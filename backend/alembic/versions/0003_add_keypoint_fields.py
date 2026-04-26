"""gloss_motions에 keypoint/video/fallback 컬럼 추가

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-22
"""
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("gloss_motions", sa.Column("keypoint_path",   sa.String(500), nullable=True))
    op.add_column("gloss_motions", sa.Column("video_path",      sa.String(500), nullable=True))
    op.add_column("gloss_motions", sa.Column("fallback_type",   sa.String(20),  nullable=True))
    op.add_column("gloss_motions", sa.Column("fallback_glosses", sa.JSON,       nullable=True))


def downgrade() -> None:
    op.drop_column("gloss_motions", "fallback_glosses")
    op.drop_column("gloss_motions", "fallback_type")
    op.drop_column("gloss_motions", "video_path")
    op.drop_column("gloss_motions", "keypoint_path")
