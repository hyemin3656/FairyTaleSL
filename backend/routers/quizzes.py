"""
quizzes 라우터

GET  /api/v1/books/{book_id}/quizzes               — 책의 모든 퀴즈 (question만 노출)
GET  /api/v1/books/{book_id}/quizzes/{section_order} — 특정 섹션 퀴즈
POST /api/v1/quizzes/{quiz_id}/check               — 답안 매칭 (정규화 후 exact match)

정답/동의어는 클라이언트에 직접 노출하지 않는다. 매칭은 서버에서 수행하고,
정답보기/최종 공개 시점에만 expected_answer를 응답에 포함한다.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from schemas.quiz import QuizOut, QuizCheckRequest, QuizCheckResponse
from services.quiz_service import (
    get_quizzes_by_book, get_quiz_by_book_section, get_quiz_by_id, is_correct,
)

router = APIRouter(tags=["quizzes"])


@router.get("/books/{book_id}/quizzes", response_model=list[QuizOut])
async def list_quizzes(book_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    return await get_quizzes_by_book(db, book_id)


@router.get("/books/{book_id}/quizzes/{section_order}", response_model=QuizOut)
async def get_section_quiz(
    book_id: uuid.UUID,
    section_order: int,
    db: AsyncSession = Depends(get_db),
):
    quiz = await get_quiz_by_book_section(db, book_id, section_order)
    if not quiz:
        raise HTTPException(status_code=404, detail="quiz not found")
    return quiz


@router.post("/quizzes/{quiz_id}/check", response_model=QuizCheckResponse)
async def check_answer(
    quiz_id: uuid.UUID,
    req: QuizCheckRequest,
    reveal: bool = Query(False, description="true면 결과와 함께 expected_answer를 반환 (정답보기/마지막 시도)"),
    db: AsyncSession = Depends(get_db),
):
    quiz = await get_quiz_by_id(db, quiz_id)
    if not quiz:
        raise HTTPException(status_code=404, detail="quiz not found")
    correct = is_correct(req.user_answer, quiz.expected_answer, quiz.synonyms)
    return QuizCheckResponse(
        correct=correct,
        expected_answer=quiz.expected_answer if (reveal or correct) else None,
    )
