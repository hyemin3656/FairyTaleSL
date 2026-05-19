"""
WebSocket — 수어 인식 엔드포인트
클라이언트(MediaPipe 랜드마크) → 서버 프레임 버퍼 누적 → AI Engine /predict → 결과 반환.

프로토콜:
  클라이언트 → {"type":"landmarks", "num_hands":N, "hands":[...]}
  서버 →       {"type":"result", "gloss":"안녕", "confidence":0.87, "is_random_init":bool}
               {"type":"idle"}
               {"type":"error", "message":"..."}
"""
import json
import random
from collections import deque

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import settings

router = APIRouter(tags=["recognition"])

# ST-GCN T_fixed=105와 맞춤 — 최대 105프레임 누적 (약 3.5초 @ ~30fps MediaPipe)
_BUFFER_SIZE = 105
_MIN_FRAMES = 10   # 최소 이 프레임 수 이상일 때만 AI Engine 호출

_DUMMY_GLOSSES = ["안녕", "감사하다", "미안하다", "좋다", "나", "너", "우리", "먹다", "보다"]


def _dummy_recognize(hands: list[dict]) -> dict:
    try:
        lm = hands[0]["landmarks"]
        avg_y = sum(p["y"] for p in lm) / len(lm)
        idx = int(avg_y * len(_DUMMY_GLOSSES)) % len(_DUMMY_GLOSSES)
        confidence = round(0.4 + random.uniform(0, 0.2), 2)
        return {"type": "result", "gloss": _DUMMY_GLOSSES[idx],
                "confidence": confidence, "is_random_init": True}
    except (KeyError, IndexError, ZeroDivisionError):
        return {"type": "idle"}


async def _call_ai_engine(hands: list[dict],
                          frame_buffer: list[dict]) -> dict | None:
    """AI Engine /predict 호출. 실패 시 None 반환."""
    payload = {"hands": hands, "frame_buffer": frame_buffer}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(
                f"{settings.AI_ENGINE_URL}/predict",
                json=payload,
            )
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "type": "result",
                    "gloss": data.get("gloss", ""),
                    "confidence": data.get("confidence", 0.0),
                    "is_random_init": data.get("is_random_init", True),
                }
    except Exception:
        pass
    return None


@router.websocket("/ws/recognition")
async def recognition_ws(websocket: WebSocket):
    await websocket.accept()

    # 연결별 프레임 버퍼 (서버 측 누적 — 프론트엔드가 버퍼 관리 불필요)
    frame_buf: deque[dict] = deque(maxlen=_BUFFER_SIZE)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "JSON 파싱 오류"})
                continue

            if msg.get("type") != "landmarks":
                continue

            hands: list[dict] = msg.get("hands", [])
            num_hands: int = msg.get("num_hands", len(hands))

            if num_hands == 0 or not hands:
                await websocket.send_json({"type": "idle"})
                continue

            # 현재 프레임을 버퍼에 추가
            frame_buf.append({"hands": hands})

            if len(frame_buf) < _MIN_FRAMES:
                # 아직 프레임이 부족 — idle 응답
                await websocket.send_json({"type": "idle"})
                continue

            result = await _call_ai_engine(hands, list(frame_buf))
            if result is None:
                result = _dummy_recognize(hands)

            await websocket.send_json(result)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
