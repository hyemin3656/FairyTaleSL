"""
Legacy recognition WebSocket endpoint.

The active follow-along UI uses local_sign_service on port 8002. This endpoint is
kept for compatibility and returns dummy recognition results.
"""
import json
import random
from collections import deque

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["recognition"])

_BUFFER_SIZE = 105
_MIN_FRAMES = 10

_DUMMY_GLOSSES = [
    "안녕",
    "감사하다",
    "미안하다",
    "좋다",
    "나",
    "너",
    "우리",
    "먹다",
    "보다",
]


def _dummy_recognize(hands: list[dict]) -> dict:
    try:
        lm = hands[0]["landmarks"]
        avg_y = sum(p["y"] for p in lm) / len(lm)
        idx = int(avg_y * len(_DUMMY_GLOSSES)) % len(_DUMMY_GLOSSES)
        confidence = round(0.4 + random.uniform(0, 0.2), 2)
        return {
            "type": "result",
            "gloss": _DUMMY_GLOSSES[idx],
            "confidence": confidence,
            "is_random_init": True,
        }
    except (KeyError, IndexError, ZeroDivisionError):
        return {"type": "idle"}


@router.websocket("/ws/recognition")
async def recognition_ws(websocket: WebSocket):
    await websocket.accept()
    frame_buf: deque[dict] = deque(maxlen=_BUFFER_SIZE)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "JSON parsing error"})
                continue

            if msg.get("type") != "landmarks":
                continue

            hands: list[dict] = msg.get("hands", [])
            num_hands: int = msg.get("num_hands", len(hands))

            if num_hands == 0 or not hands:
                await websocket.send_json({"type": "idle"})
                continue

            frame_buf.append({"hands": hands})

            if len(frame_buf) < _MIN_FRAMES:
                await websocket.send_json({"type": "idle"})
                continue

            await websocket.send_json(_dummy_recognize(hands))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
