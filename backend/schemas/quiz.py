import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class QuizOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    book_id: uuid.UUID
    section_order: int
    question: str
    # 사전 합성된 정답/동의어는 클라이언트에 노출하지 않는다 — 매칭은 서버 책임


class QuizCheckRequest(BaseModel):
    user_answer: str


class QuizCheckResponse(BaseModel):
    correct: bool
    expected_answer: str | None = None   # 오답 시 결과 화면에 표시 (정답보기 효과)
