import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class GlossMotionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    gloss: str
    gltf_clip_url: str
    emotion_label: str
    blendshape_params: dict | None
    duration_sec: float
    description: str | None
    created_at: datetime


class GlossConvertRequest(BaseModel):
    text: str  # 원문 텍스트 (한 섹션 전체)


class MotionClip(BaseModel):
    """글로스 하나에 대응하는 모션 클립 정보"""
    gloss: str
    gltf_clip_url: str
    emotion_label: str
    blendshape_params: dict
    duration_sec: float
    is_fallback: bool = False
    fallback_type: str | None = None  # "decompose" | "text" | None
    text_display: str | None = None
    # N×225 keypoint 시퀀스 (15fps): lhand(63) + rhand(63) + pose(99)
    keypoints: list[list[float]] | None = None
    fps: float = 15.0


class GlossConvertResponse(BaseModel):
    original_text: str
    tokens: list[str]          # 토크나이징 결과
    clips: list[MotionClip]    # 글로스별 모션 클립 순서대로
    total_duration_sec: float
