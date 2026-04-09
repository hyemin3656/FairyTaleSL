import uuid
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=2, max_length=50)
    nickname: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=6)
    birthdate: date | None = None
    gender: str | None = None  # "M" | "F" | "OTHER"
    role: str = "child"


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    username: str
    nickname: str
    role: str
    avatar_speed: float
    subtitle_enabled: bool
    created_at: datetime


# ── 학습 이력 ──────────────────────────────────────────────────────────────────
class SessionQAOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    section_order: int
    question_text: str
    user_answer_text: str | None
    llm_response_text: str | None
    recognition_accuracy: float | None
    created_at: datetime


class LearningSessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    book_id: uuid.UUID | None
    book_title: str | None = None        # join으로 채움
    last_section_order: int
    status: str
    avg_recognition_accuracy: float | None
    started_at: datetime
    ended_at: datetime | None
    qa_records: list[SessionQAOut] = []


class MyPageResponse(BaseModel):
    user: UserOut
    sessions: list[LearningSessionOut]
    total_sessions: int
    completed_sessions: int
    avg_score: float | None  # 전체 Q&A 평균 점수
