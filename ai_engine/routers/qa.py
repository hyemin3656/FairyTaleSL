"""
POST /qa/question — T5 섹션 이해도 질문 생성 (레거시)
POST /qa/evaluate — T5 답변 평가 (레거시)
POST /qa/followup — T5 후속 질문 (레거시)
POST /qa/child    — Gemini 2.5 Flash 아이 자유질문 응답 (FairyTaleSL 시나리오)
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.t5_qa import generate_question, evaluate_answer, generate_followup_question
from models.gemini_child import answer_child_question

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


class ChildQARequest(BaseModel):
    question: str
    story_context: str   # 현재 섹션 본문 (TODO: 멀티섹션 컨텍스트 도입 시 string concat)


@router.post("/child")
async def qa_child(req: ChildQARequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    try:
        answer = answer_child_question(req.question, req.story_context)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower() or " 429" in msg:
            raise HTTPException(
                status_code=429,
                detail="오늘의 AI 응답 한도를 모두 사용했어요. 잠시 후 다시 시도해 주세요.",
            )
        raise HTTPException(status_code=500, detail=f"gemini error: {msg}")
    return {"answer": answer}
