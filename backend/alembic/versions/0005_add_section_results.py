"""section_results 테이블 추가

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-13
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "section_results",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id", sa.Uuid(as_uuid=True),
            sa.ForeignKey("learning_sessions.id", ondelete="CASCADE"),
            nullable=False, index=True,
        ),
        sa.Column("section_order", sa.Integer, nullable=False),
        sa.Column("follow_along_passed", sa.Boolean, nullable=True),
        sa.Column("quiz_correct", sa.Boolean, nullable=True),
        sa.Column("quiz_attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("session_id", "section_order", name="uq_section_result_session_order"),
    )


def downgrade() -> None:
    op.drop_table("section_results")
