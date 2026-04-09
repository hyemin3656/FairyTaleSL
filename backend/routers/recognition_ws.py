"""
WebSocket — 수어 인식 엔드포인트
클라이언트(MediaPipe 랜드마크) → AI Engine /predict → 결과 반환.
AI Engine 미응답 시 더미 fallback.

프로토콜:
  클라이언트 → {"type":"landmarks", "num_hands":N, "hands":[...], "frame_buffer":[...]}
  서버 →       {"type":"result",  "gloss":"토끼", "confidence":0.87, "is_random_init":bool}
               {"type":"idle"}
               {"type":"error", "message":"..."}
"""
import json
import random

import httpx
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.config import settings

router = APIRouter(tags=["recognition"])

_DUMMY_GLOSSES = ["토끼", "거북이", "경주", "달나라", "별", "도깨비", "할아버지", "소녀", "우주"]


def _dummy_recognize(hands: list[dict]) -> dict:
    try:
        lm = hands[0]["landmarks"]
        avg_y = sum(p["y"] for p in lm) / len(lm)
        idx = int(avg_y * len(_DUMMY_GLOSSES)) % len(_DUMMY_GLOSSES)
        confidence = round(0.55 + (1 - avg_y) * 0.35 + random.uniform(-0.05, 0.05), 2)
        confidence = max(0.5, min(0.99, confidence))
        return {"type": "result", "gloss": _DUMMY_GLOSSES[idx],
                "confidence": confidence, "is_random_init": True}
    except (KeyError, IndexError, ZeroDivisionError):
        return {"type": "idle"}


async def _call_ai_engine(hands: list[dict], frame_buffer: list | None) -> dict | None:
    """AI Engine /predict 호출. 실패 시 None 반환."""
    payload = {"hands": hands}
    if frame_buffer:
        payload["frame_buffer"] = frame_buffer
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
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

            num_hands = msg.get("num_hands", 0)
            hands = msg.get("hands", [])
            frame_buffer = msg.get("frame_buffer")

            if num_hands == 0 or not hands:
                await websocket.send_json({"type": "idle"})
                continue

            # AI Engine 우선 시도
            result = await _call_ai_engine(hands, frame_buffer)
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
