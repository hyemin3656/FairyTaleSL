import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BookSectionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    order: int
    title: str | None
    text: str
    sign_text: str | None = None
    image_url: str | None = None
    image_urls: list[str] | None = None
    image_triggers: list[str | None] | None = None


class BookListItem(BaseModel):
    """목록 카드용 — 구간 텍스트 제외"""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    description: str | None
    cover_image_url: str | None
    target_age_min: int
    target_age_max: int
    categories: list[str]
    author: str | None
    created_at: datetime


class BookDetail(BookListItem):
    """상세 조회 — 구간 포함"""

    sections: list[BookSectionOut]


class BookListResponse(BaseModel):
    total: int
    items: list[BookListItem]
