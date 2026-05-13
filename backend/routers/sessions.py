"""
학습 세션 라우터 (JWT 필요)
POST /sessions              — 세션 시작
PATCH /sessions/{id}        — 세션 업데이트 (진행도, 상태)
POST /sessions/{id}/qa      — QA 기록 저장
GET  /sessions/me           — 내 세션 목록 (마이페이지용)
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from routers.auth import get_current_user
from schemas.user import LearningSessionOut, MyPageResponse, UserOut, SessionQAOut
from services.user_service import (
    create_session, update_session, save_qa_record, get_user_sessions,
    upsert_section_result,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


class SessionStartRequest(BaseModel):
    book_id: uuid.UUID | None = None


class SessionUpdateRequest(BaseModel):
    last_section_order: int | None = None
    status: str | None = None                  # "in_progress" | "completed" | "abandoned"
    avg_recognition_accuracy: float | None = None


class QARecordRequest(BaseModel):
    section_order: int
    question: str
    user_answer: str | None = None
    llm_response: str | None = None
    score: float | None = None                 # 0.0–1.0


class SectionResultRequest(BaseModel):
    section_order: int
    follow_along_passed: bool | None = None
    quiz_correct: bool | None = None
    quiz_attempts: int | None = None


@router.post("", response_model=dict, status_code=201)
async def start_session(
    req: SessionStartRequest,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await create_session(db, user.id, req.book_id)
    return {"session_id": str(session.id)}


@router.patch("/{session_id}", response_model=dict)
async def patch_session(
    session_id: uuid.UUID,
    req: SessionUpdateRequest,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await update_session(
        db, session_id,
        last_section_order=req.last_section_order,
        status=req.status,
        avg_recognition_accuracy=req.avg_recognition_accuracy,
    )
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return {"ok": True}


@router.post("/{session_id}/qa", response_model=dict, status_code=201)
async def add_qa(
    session_id: uuid.UUID,
    req: QARecordRequest,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    record = await save_qa_record(
        db, session_id,
        section_order=req.section_order,
        question=req.question,
        user_answer=req.user_answer,
        llm_response=req.llm_response,
        score=req.score,
    )
    return {"qa_id": str(record.id)}


@router.post("/{session_id}/section-result", response_model=dict, status_code=201)
async def upsert_section(
    session_id: uuid.UUID,
    req: SectionResultRequest,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """섹션별 결과 upsert (따라하기/퀴즈). 같은 (session_id, section_order)는 덮어쓰기."""
    row = await upsert_section_result(
        db, session_id,
        section_order=req.section_order,
        follow_along_passed=req.follow_along_passed,
        quiz_correct=req.quiz_correct,
        quiz_attempts=req.quiz_attempts,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return {"ok": True, "id": str(row.id)}


@router.get("/me", response_model=MyPageResponse)
async def my_page(
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    sessions_raw = await get_user_sessions(db, user.id)

    sessions_out = []
    all_scores: list[float] = []

    for s in sessions_raw:
        qa_out = [SessionQAOut.model_validate(q) for q in s["qa_records"]]
        for q in s["qa_records"]:
            if q.recognition_accuracy is not None:
                all_scores.append(q.recognition_accuracy)
        sessions_out.append(LearningSessionOut(
            id=s["id"],
            book_id=s["book_id"],
            book_title=s["book_title"],
            last_section_order=s["last_section_order"],
            status=s["status"],
            avg_recognition_accuracy=s["avg_recognition_accuracy"],
            started_at=s["started_at"],
            ended_at=s["ended_at"],
            qa_records=qa_out,
        ))

    completed = sum(1 for s in sessions_raw if s["status"] == "completed")
    avg = round(sum(all_scores) / len(all_scores), 3) if all_scores else None

    return MyPageResponse(
        user=UserOut.model_validate(user),
        sessions=sessions_out,
        total_sessions=len(sessions_out),
        completed_sessions=completed,
        avg_score=avg,
    )
