"""
WebSocket — 글로스 스트리밍 엔드포인트

메시지 프로토콜 (클라이언트→서버):
  {"text": "...", "start_index"?: N}   텍스트 전송 (start_index부터 스트리밍, 기본 0)
  {"type": "pause"}                    일시정지
  {"type": "resume"}                   재개

메시지 프로토콜 (서버→클라이언트):
  {"type": "start",  "total": N, "tokens": [...]}
  {"type": "clip",   "index": i, "clip": MotionClip}   ← i는 전체 토큰 기준 절대 인덱스
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

    async def do_stream(clips, start_index: int = 0):
        """clips를 순서대로 전송. pause_event 대기로 일시정지 지원.
        index는 전체 토큰 기준 절대값(start_index + i)으로 전송."""
        try:
            for i, clip in enumerate(clips):
                await websocket.send_json(
                    {"type": "clip", "index": start_index + i, "clip": clip.model_dump()}
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

            # start_index: 동화 본문을 일시정지 시점부터 다시 재생할 때 사용 (기본 0)
            try:
                start_index = max(0, int(msg.get("start_index", 0) or 0))
            except (TypeError, ValueError):
                start_index = 0

            # 진행 중인 스트림 취소
            if stream_task and not stream_task.done():
                stream_task.cancel()
                await asyncio.gather(stream_task, return_exceptions=True)

            # 새 스트림 시작 시 pause 해제
            pause_event.set()

            tokens = tokenize_text(text)
            async with AsyncSessionLocal() as db:
                clips = await resolve_motions(db, tokens)

            # start_index가 토큰 수를 넘으면 0으로 클램프
            if start_index >= len(clips):
                start_index = 0

            await websocket.send_json(
                {"type": "start", "total": len(clips), "tokens": [c.gloss for c in clips]}
            )
            stream_task = asyncio.create_task(
                do_stream(clips[start_index:], start_index=start_index)
            )

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
