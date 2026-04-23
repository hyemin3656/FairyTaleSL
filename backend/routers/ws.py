"""
WebSocket — 글로스 스트리밍 엔드포인트

메시지 프로토콜 (클라이언트→서버):
  {"text": "..."}            텍스트 전송 (스트리밍 시작)
  {"type": "pause"}          일시정지
  {"type": "resume"}         재개

메시지 프로토콜 (서버→클라이언트):
  {"type": "start",  "total": N, "tokens": [...]}
  {"type": "clip",   "index": i, "clip": MotionClip}
  {"type": "done"}
  {"type": "paused"}
  {"type": "resumed"}
  {"type": "error",  "message": "..."}
"""
import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.database import AsyncSessionLocal
from services.gloss_service import tokenize_text, resolve_motions

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/gloss")
async def gloss_ws(websocket: WebSocket):
    await websocket.accept()

    pause_event = asyncio.Event()
    pause_event.set()          # set = 재생 중, clear = 일시정지
    stream_task: asyncio.Task | None = None

    async def do_stream(clips):
        """클립을 순서대로 전송. pause_event 대기로 일시정지 지원."""
        try:
            for i, clip in enumerate(clips):
                await websocket.send_json(
                    {"type": "clip", "index": i, "clip": clip.model_dump()}
                )
                # duration_sec 동안 대기 (0.05s 단위로 pause 체크)
                remaining = clip.duration_sec
                step = 0.05
                while remaining > 0:
                    await pause_event.wait()   # 일시정지 중이면 여기서 블록
                    sleep_time = min(step, remaining)
                    await asyncio.sleep(sleep_time)
                    remaining -= sleep_time
            await websocket.send_json({"type": "done"})
        except asyncio.CancelledError:
            pass

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, AttributeError):
                await websocket.send_json(
                    {"type": "error", "message": "올바른 JSON 형식이 아닙니다."}
                )
                continue

            msg_type = msg.get("type", "")

            # ── 일시정지 / 재개 ───────────────────────────────────
            if msg_type == "pause":
                pause_event.clear()
                await websocket.send_json({"type": "paused"})
                continue

            if msg_type == "resume":
                pause_event.set()
                await websocket.send_json({"type": "resumed"})
                continue

            # ── 텍스트 수신 → 스트리밍 시작 ──────────────────────
            text = msg.get("text", "").strip()
            if not text:
                await websocket.send_json(
                    {"type": "error", "message": "text가 비어 있습니다."}
                )
                continue

            # 진행 중인 스트림 취소
            if stream_task and not stream_task.done():
                stream_task.cancel()
                await asyncio.gather(stream_task, return_exceptions=True)

            # 새 스트림 시작 시 pause 해제
            pause_event.set()

            tokens = tokenize_text(text)
            async with AsyncSessionLocal() as db:
                clips = await resolve_motions(db, tokens)

            await websocket.send_json(
                {"type": "start", "total": len(clips), "tokens": tokens}
            )
            stream_task = asyncio.create_task(do_stream(clips))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        if stream_task:
            stream_task.cancel()
