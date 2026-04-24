import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column

from core.database import Base


class GlossMotion(Base):
    """글로스 → GLTF 애니메이션 클립 매핑 테이블 (Motion DB)"""

    __tablename__ = "gloss_motions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    gloss: Mapped[str] = mapped_column(
        String(100), unique=True, nullable=False, index=True
    )
    # GLTF 클립 파일 경로 또는 CDN URL
    gltf_clip_url: Mapped[str] = mapped_column(String(500), nullable=False)
    # 감정 레이블: neutral / happy / sad / surprised / angry
    emotion_label: Mapped[str] = mapped_column(String(30), default="neutral")
    # 블렌드셰이프 파라미터 JSON { "eyeBlinkLeft": 0.0, ... }
    blendshape_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    duration_sec: Mapped[float] = mapped_column(Float, default=1.0)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    keypoint_path:   Mapped[str | None] = mapped_column(String(500), nullable=True)
    video_path:      Mapped[str | None] = mapped_column(String(500), nullable=True)
    fallback_type:   Mapped[str | None] = mapped_column(String(20),  nullable=True)
    fallback_glosses: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
