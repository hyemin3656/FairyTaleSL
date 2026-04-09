"""사용자 및 학습 세션 서비스"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.security import hash_password, verify_password
from models.user import User
from models.session import LearningSession, SessionQA
from models.book import Book
from schemas.user import RegisterRequest


async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    result = await db.execute(select(User).where(User.username == username))
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, req: RegisterRequest) -> User:
    user = User(
        email=req.email,
        username=req.username,
        nickname=req.nickname,
        hashed_password=hash_password(req.password),
        birthdate=req.birthdate,
        gender=req.gender,
        role=req.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


async def authenticate(db: AsyncSession, username: str, password: str) -> User | None:
    user = await get_user_by_username(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


# ── 학습 세션 ──────────────────────────────────────────────────────────────────
async def create_session(db: AsyncSession, user_id: uuid.UUID,
                         book_id: uuid.UUID | None) -> LearningSession:
    session = LearningSession(user_id=user_id, book_id=book_id)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def update_session(db: AsyncSession, session_id: uuid.UUID,
                         last_section_order: int | None = None,
                         status: str | None = None,
                         avg_recognition_accuracy: float | None = None) -> LearningSession | None:
    result = await db.execute(
        select(LearningSession).where(LearningSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        return None
    if last_section_order is not None:
        session.last_section_order = last_section_order
    if status is not None:
        session.status = status
        if status == "completed":
            session.ended_at = datetime.now(timezone.utc)
    if avg_recognition_accuracy is not None:
        session.avg_recognition_accuracy = avg_recognition_accuracy
    await db.commit()
    await db.refresh(session)
    return session


async def save_qa_record(db: AsyncSession, session_id: uuid.UUID,
                         section_order: int, question: str,
                         user_answer: str | None, llm_response: str | None,
                         score: float | None) -> SessionQA:
    record = SessionQA(
        session_id=session_id,
        section_order=section_order,
        question_text=question,
        user_answer_text=user_answer,
        llm_response_text=llm_response,
        recognition_accuracy=score,
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


async def get_user_sessions(db: AsyncSession, user_id: uuid.UUID) -> list:
    """유저의 세션 목록 + QA 기록 + 책 제목 join."""
    result = await db.execute(
        select(LearningSession)
        .where(LearningSession.user_id == user_id)
        .options(selectinload(LearningSession.qa_records))
        .order_by(LearningSession.started_at.desc())
    )
    sessions = result.scalars().all()

    # 책 제목 보완
    book_ids = {s.book_id for s in sessions if s.book_id}
    book_map: dict[uuid.UUID, str] = {}
    if book_ids:
        books_result = await db.execute(
            select(Book.id, Book.title).where(Book.id.in_(book_ids))
        )
        book_map = {row.id: row.title for row in books_result}

    out = []
    for s in sessions:
        d = {
            "id": s.id,
            "book_id": s.book_id,
            "book_title": book_map.get(s.book_id) if s.book_id else None,
            "last_section_order": s.last_section_order,
            "status": s.status,
            "avg_recognition_accuracy": s.avg_recognition_accuracy,
            "started_at": s.started_at,
            "ended_at": s.ended_at,
            "qa_records": s.qa_records,
        }
        out.append(d)
    return out
