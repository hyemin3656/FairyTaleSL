"""SectionResult — 학습 세션 내 섹션별 결과 (따라하기 통과 / 퀴즈 정/오답·시도횟수)"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import Uuid

from core.database import Base


class SectionResult(Base):
    __tablename__ = "section_results"
    __table_args__ = (
        UniqueConstraint("session_id", "section_order", name="uq_section_result_session_order"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    section_order: Mapped[int] = mapped_column(Integer, nullable=False)

    follow_along_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    quiz_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    quiz_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    session: Mapped["LearningSession"] = relationship()  # noqa: F821
