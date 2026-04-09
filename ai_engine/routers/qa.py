"""
POST /qa/question — 질문 생성
POST /qa/evaluate — 답변 평가
"""
from fastapi import APIRouter
from pydantic import BaseModel

from models.t5_qa import generate_question, evaluate_answer, generate_followup_question

router = APIRouter(prefix="/qa", tags=["qa"])


class QuestionRequest(BaseModel):
    section_text: str


class EvaluateRequest(BaseModel):
    question: str
    context: str       # 섹션 원문
    user_answer: str


@router.post("/question")
async def make_question(req: QuestionRequest):
    question = generate_question(req.section_text)
    return {"question": question}


@router.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    result = evaluate_answer(
        question=req.question,
        context=req.context,
        user_answer=req.user_answer,
    )
    return result


class FollowupRequest(BaseModel):
    context: str
    prev_question: str
    user_answer: str
    feedback: str


@router.post("/followup")
async def followup(req: FollowupRequest):
    question = generate_followup_question(
        context=req.context,
        prev_question=req.prev_question,
        user_answer=req.user_answer,
        feedback=req.feedback,
    )
    return {"question": question}
