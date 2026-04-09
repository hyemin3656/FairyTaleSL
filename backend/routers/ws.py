"""
WebSocket — 글로스 스트리밍 엔드포인트
클라이언트가 텍스트를 보내면 글로스별 모션 클립을 duration_sec 간격으로 스트리밍.

메시지 프로토콜 (서버→클라이언트):
  {"type": "start",  "total": N, "tokens": [...]}
  {"type": "clip",   "index": i, "clip": MotionClip}
  {"type": "done"}
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
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
                text = payload.get("text", "").strip()
            except (json.JSONDecodeError, AttributeError):
                await websocket.send_json(
                    {"type": "error", "message": "올바른 JSON 형식이 아닙니다."}
                )
                continue

            if not text:
                await websocket.send_json(
                    {"type": "error", "message": "text가 비어 있습니다."}
                )
                continue

            tokens = tokenize_text(text)
            async with AsyncSessionLocal() as db:
                clips = await resolve_motions(db, tokens)

            await websocket.send_json(
                {"type": "start", "total": len(clips), "tokens": tokens}
            )

            for i, clip in enumerate(clips):
                await websocket.send_json(
                    {"type": "clip", "index": i, "clip": clip.model_dump()}
                )
                await asyncio.sleep(clip.duration_sec)

            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
