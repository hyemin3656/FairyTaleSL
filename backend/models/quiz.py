"""
Quiz — 사전 생성된 섹션별 단답형 퀴즈 (Gemini 2.5 Flash 출력)

배치 파이프라인:
  data_pipeline/quiz_generation/generate_quizzes.py  → JSON
  backend/scripts/load_quizzes.py                    → 이 테이블 적재

런타임:
  GET  /books/{book_id}/quizzes      — 책의 전체 퀴즈
  POST /sessions/{id}/quiz_attempts  — 시도 결과 기록 (Phase 6)
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON, Uuid

from core.database import Base


class Quiz(Base):
    __tablename__ = "quizzes"
    __table_args__ = (
        UniqueConstraint("book_id", "section_order", name="uq_quiz_book_section"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    book_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=False, index=True
    )
    section_order: Mapped[int] = mapped_column(Integer, nullable=False)

    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(String(200), nullable=False)
    # Gemini가 생성한 동의어/유사 표현 풀 — 답안 매칭 시 expected_answer와 함께 비교
    synonyms: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
