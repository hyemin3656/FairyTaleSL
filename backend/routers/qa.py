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
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="AI 엔진에 연결할 수 없습니다.")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="AI 엔진 응답이 너무 늦어요.")

    if resp.status_code >= 400:
        # AI 엔진의 status code와 detail을 그대로 통과시켜 프론트가 의미 있는 메시지를 보이게.
        try:
            detail = resp.json().get("detail") or resp.text
        except Exception:
            detail = resp.text or "AI 엔진 오류"
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return resp.json()


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


class ChildQARequest(BaseModel):
    question: str
    story_context: str


@router.post("/child")
async def child_qa(req: ChildQARequest):
    return await _post("/qa/child", req.model_dump())
