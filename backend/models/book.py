import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON, Uuid

from core.database import Base


class Book(Base):
    """동화책 메타데이터"""

    __tablename__ = "books"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    cover_image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    target_age_min: Mapped[int] = mapped_column(Integer, default=5)
    target_age_max: Mapped[int] = mapped_column(Integer, default=10)
    # 카테고리: 동물, 일상, 전래민담 등 — 복수 허용 (OR 조건 검색)
    # JSON 타입 → SQLite(테스트)·PostgreSQL(운영) 모두 호환
    categories: Mapped[list[str]] = mapped_column(
        JSON, nullable=False, default=list
    )
    author: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    sections: Mapped[list["BookSection"]] = relationship(
        back_populates="book",
        cascade="all, delete-orphan",
        order_by="BookSection.order",
    )
    sessions: Mapped[list["LearningSession"]] = relationship(  # noqa: F821
        back_populates="book"
    )


class BookSection(Base):
    """동화 구간 — 아바타 재생 및 질문 생성 단위"""

    __tablename__ = "book_sections"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    book_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("books.id", ondelete="CASCADE"), nullable=False
    )
    order: Mapped[int] = mapped_column(Integer, nullable=False)  # 구간 순서 (1-based)
    title: Mapped[str | None] = mapped_column(String(200), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)  # 원문 텍스트
    image_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    image_urls: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    image_triggers: Mapped[list[str | None] | None] = mapped_column(ARRAY(Text), nullable=True)

    book: Mapped["Book"] = relationship(back_populates="sections")


# 카테고리 상수 — API 응답·필터에 사용
class BookCategory:
    ANIMAL = "동물"
    DAILY = "일상"
    FOLKTALE = "전래민담"
    FANTASY = "판타지"
    SCIENCE = "과학"
    FRIENDSHIP = "우정"

    ALL = [ANIMAL, DAILY, FOLKTALE, FANTASY, SCIENCE, FRIENDSHIP]
