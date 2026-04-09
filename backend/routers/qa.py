"""
Q&A 프록시 라우터 — AI Engine /qa/* 를 프론트엔드에 노출.
"""
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.config import settings

router = APIRouter(prefix="/qa", tags=["qa"])


class QuestionRequest(BaseModel):
    section_text: str


class EvaluateRequest(BaseModel):
    question: str
    context: str
    user_answer: str


async def _post(path: str, body: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{settings.AI_ENGINE_URL}{path}", json=body)
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="AI 엔진에 연결할 수 없습니다.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/question")
async def make_question(req: QuestionRequest):
    return await _post("/qa/question", {"section_text": req.section_text})


@router.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    return await _post("/qa/evaluate", req.model_dump())


class FollowupRequest(BaseModel):
    context: str
    prev_question: str
    user_answer: str
    feedback: str


@router.post("/followup")
async def make_followup(req: FollowupRequest):
    return await _post("/qa/followup", req.model_dump())
