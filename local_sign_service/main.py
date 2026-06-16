"""
local_sign_service — 수어 인식 사이드카 (포트 8002, CPU)

설계 개요:
  - 카메라는 이 프로세스가 단독 점유 → 브라우저 getUserMedia 충돌 회피
  - MediaPipe → 키포인트 → CNN1D 추론까지 클라이언트 머신에서 완결
  - 키포인트는 서버(8000/8001)로 전송하지 않음 (박혜민 의도)
  - WebSocket /ws/sign 으로 브라우저에 push:
      * preview     : JPEG base64 미리보기 (~10fps throttle)
      * prediction  : {gloss, confidence, is_dummy}   ← main 호환 스키마
      * idle        : 손 미검출

스레드:
  Thread A  카메라 캡처 → raw_frame_q (latest-only, maxsize=2)
  Thread B  MediaPipe 처리 + Segmenter → infer_q + preview_q
  Thread C  CNN1D 추론 → result_q

asyncio 이벤트 루프 (메인):
  - preview_q, result_q를 소비해 모든 연결된 WS에 broadcast
  - 60ms 주기로 큐 poll (낮은 latency + 낮은 CPU)
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# 런타임 환경 변수 — 모델/디바이스 등을 ENV로 조정 가능 (.env 또는 export)
PROJECT_ROOT = Path(__file__).resolve().parent
DEVICE = os.environ.get("SIGN_DEVICE", "cpu")
CAMERA_INDEX = int(os.environ.get("SIGN_CAMERA", "0"))
FPS = int(os.environ.get("SIGN_FPS", "30"))
CAM_W = int(os.environ.get("SIGN_WIDTH", "640"))
CAM_H = int(os.environ.get("SIGN_HEIGHT", "480"))
PREVIEW_FPS = int(os.environ.get("SIGN_PREVIEW_FPS", "10"))
JPEG_QUALITY = int(os.environ.get("SIGN_JPEG_QUALITY", "70"))
DRAW_KEYPOINTS = os.environ.get("SIGN_DRAW_KEYPOINTS", "1").lower() not in {"0", "false", "no", "off"}
MIRROR_FRAME = os.environ.get("SIGN_MIRROR", "0").lower() in {"1", "true", "yes", "on"}
MODE_DEFAULT = os.environ.get("SIGN_MODE", "gloss")   # "gloss" | "sequence"

CONFIG_PATH = Path(os.environ.get(
    "SIGN_CONFIG",
    PROJECT_ROOT / "src" / "cnn1d_config.py",
))
CHECKPOINT_PATH = Path(os.environ.get(
    "SIGN_CHECKPOINT",
    PROJECT_ROOT / "checkpoints" / "best.pth",
))
LABEL_MAP_PATH = Path(os.environ.get("SIGN_LABEL_MAP", PROJECT_ROOT / "src" / "class_labels.json"))
TOP1_THRESHOLD = float(os.environ.get("SIGN_TOP1_THRESHOLD", "0.2"))

log = logging.getLogger("sign-sidecar")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── 글로벌 상태 ────────────────────────────────────────────────────────
# asyncio는 단일 이벤트 루프에서 동작하므로 set의 add/discard는 await 없이도 thread-safe.
# 별도 Lock을 두지 않아 이벤트루프 바인딩 이슈를 피한다.
_clients: set[WebSocket] = set()
_state = {"mode": MODE_DEFAULT, "running": False}


def _state_snapshot() -> dict:
    return {"mode": _state["mode"], "running": _state["running"], "fps": FPS}


# ── 카메라/추론 스레드 ─────────────────────────────────────────────────
class SignWorker:
    """카메라 → MediaPipe → Segmenter → CNN1D 파이프라인을 백그라운드에서 운영."""

    def __init__(self) -> None:
        self.raw_q: queue.Queue = queue.Queue(maxsize=2)     # latest-only
        self.infer_q: queue.Queue = queue.Queue(maxsize=4)
        self.preview_q: queue.Queue = queue.Queue(maxsize=2)
        self.result_q: queue.Queue = queue.Queue(maxsize=16)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.mode = MODE_DEFAULT
        self.recognizer = None
        self.extractor = None
        self.segmenter = None

    def set_mode(self, mode: str) -> None:
        if mode in ("gloss", "sequence"):
            self.mode = mode
            if self.segmenter is not None:
                self.segmenter.sequence_level = (mode == "sequence")
                self.segmenter.reset()
                log.info("mode switched to %s", mode)

    def start(self) -> None:
        if self._threads:
            return
        from recognizer import (
            MediaPipeWebcamExtractor, RealtimeSegmenter, CNN1DRealtimeRecognizer
        )
        log.info("loading CNN1D model on %s …", DEVICE)
        self.recognizer = CNN1DRealtimeRecognizer(
            config=CONFIG_PATH,
            checkpoint=CHECKPOINT_PATH,
            label_map=LABEL_MAP_PATH,
            device=DEVICE,
            topk=5,
        )
        self.extractor = MediaPipeWebcamExtractor()
        self.segmenter = RealtimeSegmenter(
            fps=FPS,
            sequence_level=(self.mode == "sequence"),
        )
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True, name="cap"),
            threading.Thread(target=self._extract_loop, daemon=True, name="ext"),
            threading.Thread(target=self._infer_loop, daemon=True, name="inf"),
        ]
        for t in self._threads:
            t.start()
        _state["running"] = True
        log.info("worker started")

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        self._threads = []
        if self.extractor is not None:
            self.extractor.close()
        _state["running"] = False
        log.info("worker stopped")

    # ── Thread A : 카메라 캡처 ─────────────────────────────────────────
    def _capture_loop(self) -> None:
        import cv2
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        cap.set(cv2.CAP_PROP_FPS, FPS)
        log.info("camera opened idx=%d %dx%d@%d", CAMERA_INDEX, CAM_W, CAM_H, FPS)
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                captured_at = time.perf_counter()
                if not ok:
                    time.sleep(0.01)
                    continue
                if MIRROR_FRAME:
                    frame = cv2.flip(frame, 1)  # 미러
                item = (frame, captured_at)
                # latest-only: 이전 프레임 버리고 최신만
                try:
                    self.raw_q.put_nowait(item)
                except queue.Full:
                    try:
                        self.raw_q.get_nowait()
                    except queue.Empty:
                        pass
                    self.raw_q.put_nowait(item)
        finally:
            cap.release()

    # ── Thread B : MediaPipe + Segmenter + JPEG 인코딩 ─────────────────
    def _extract_loop(self) -> None:
        import cv2
        if DRAW_KEYPOINTS:
            from recognizer import draw_keypoints
        else:
            draw_keypoints = None

        last_preview_ts = 0.0
        preview_interval = 1.0 / max(1, PREVIEW_FPS)
        while not self._stop.is_set():
            try:
                frame_item = self.raw_q.get(timeout=0.1)
            except queue.Empty:
                continue
            frame, captured_at = frame_item
            data = self.extractor.process(frame)
            data["captured_at"] = captured_at
            if draw_keypoints is not None:
                draw_keypoints(frame, data)
            data["_frame_bgr"] = frame  # preview용

            # segment 결과 → infer_q
            for prod in self.segmenter.update(data):
                try:
                    self.infer_q.put_nowait(prod)
                except queue.Full:
                    pass

            # preview throttle
            now = time.perf_counter()
            if (now - last_preview_ts) >= preview_interval:
                last_preview_ts = now
                jpg_bytes = self._encode_jpeg(frame, data)
                try:
                    self.preview_q.put_nowait(jpg_bytes)
                except queue.Full:
                    try:
                        self.preview_q.get_nowait()
                    except queue.Empty:
                        pass
                    self.preview_q.put_nowait(jpg_bytes)

    @staticmethod
    def _encode_jpeg(frame, data) -> bytes:
        import cv2
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return b""
        return buf.tobytes()

    # ── Thread C : 추론 ────────────────────────────────────────────────
    def _infer_loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.infer_q.get(timeout=0.1)
            except queue.Empty:
                continue
            segment = item["segment"]
            seg_type = item["type"]   # "final" | "window"
            t0 = time.perf_counter()
            try:
                out = self.recognizer.predict_segment(segment)
            except Exception as e:
                log.exception("predict failed: %s", e)
                continue
            elapsed_ms = (time.perf_counter() - t0) * 1000
            top1 = out["top1"]
            if not top1 or top1["score"] < TOP1_THRESHOLD:
                continue
            try:
                self.result_q.put_nowait({
                    "type": "prediction",
                    "gloss": top1["label"],
                    "confidence": top1["score"],
                    "is_dummy": False,
                    "seg_type": seg_type,
                    "inference_ms": round(elapsed_ms, 2),
                    "topk": out["predictions"],
                })
            except queue.Full:
                pass


# ── 단일 worker 인스턴스 ───────────────────────────────────────────────
_worker = SignWorker()


# ── asyncio: 큐 폴링 + 브로드캐스트 ────────────────────────────────────
async def _broadcast(msg: dict) -> None:
    if not _clients:
        return
    dead: list[WebSocket] = []
    for ws in list(_clients):   # 반복 중 set 변경 안전
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


async def _preview_broadcast(jpeg_bytes: bytes) -> None:
    if not jpeg_bytes:
        return
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    await _broadcast({"type": "preview", "jpeg_b64": b64})


async def _poll_queues_forever() -> None:
    while True:
        sent = False
        # preview
        try:
            jpg = _worker.preview_q.get_nowait()
            await _preview_broadcast(jpg)
            sent = True
        except queue.Empty:
            pass
        # result
        try:
            res = _worker.result_q.get_nowait()
            await _broadcast(res)
            sent = True
        except queue.Empty:
            pass
        # idle hint (segment 종료 후 ~)
        if not sent:
            await asyncio.sleep(0.03)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _worker.start()
    poll_task = asyncio.create_task(_poll_queues_forever())
    log.info("sidecar ready")
    yield
    poll_task.cancel()
    _worker.stop()


# ── FastAPI 앱 ─────────────────────────────────────────────────────────
app = FastAPI(title="FairyTaleSL Local Sign Service", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "service": "local_sign_service", **_state_snapshot()}


@app.get("/state")
async def state() -> dict:
    return _state_snapshot()


@app.websocket("/ws/sign")
async def ws_sign(ws: WebSocket) -> None:
    client = f"{ws.client.host}:{ws.client.port}" if ws.client else "unknown"
    try:
        await ws.accept()
    except Exception as e:
        log.error("ws accept failed (%s): %s", client, e)
        return
    log.info("ws connected: %s (clients=%d)", client, len(_clients) + 1)
    _clients.add(ws)
    try:
        await ws.send_json({"type": "ready", **_state_snapshot()})
    except Exception as e:
        log.error("ws send ready failed (%s): %s", client, e)
        _clients.discard(ws)
        return

    import json
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")
            if mtype == "set_mode":
                _worker.set_mode(msg.get("mode", "gloss"))
                _state["mode"] = _worker.mode
                await _broadcast({"type": "state", **_state_snapshot()})
            elif mtype == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        log.info("ws disconnected: %s", client)
    except Exception as e:
        log.exception("ws error (%s): %s", client, e)
    finally:
        _clients.discard(ws)
