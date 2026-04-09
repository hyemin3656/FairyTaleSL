"""add users and sessions (SQLite compatible)

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-09
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # SQLite에서는 IF NOT EXISTS 미지원 → try/except
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = inspector.get_table_names()

    if "users" not in existing:
        op.create_table(
            "users",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("email", sa.String(255), nullable=False, unique=True),
            sa.Column("username", sa.String(50), nullable=False, unique=True),
            sa.Column("nickname", sa.String(50), nullable=False),
            sa.Column("hashed_password", sa.String(255), nullable=False),
            sa.Column("birthdate", sa.Date, nullable=True),
            sa.Column("gender", sa.String(10), nullable=True),
            sa.Column("role", sa.String(20), nullable=False, server_default="child"),
            sa.Column("avatar_speed", sa.Float, nullable=False, server_default="1.0"),
            sa.Column("subtitle_enabled", sa.Boolean, nullable=False, server_default="1"),
            sa.Column("is_active", sa.Boolean, nullable=False, server_default="1"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    if "learning_sessions" not in existing:
        op.create_table(
            "learning_sessions",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("user_id", sa.Uuid(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("book_id", sa.Uuid(as_uuid=True), sa.ForeignKey("books.id", ondelete="SET NULL"), nullable=True),
            sa.Column("last_section_order", sa.Integer, nullable=False, server_default="0"),
            sa.Column("status", sa.String(20), nullable=False, server_default="in_progress"),
            sa.Column("avg_recognition_accuracy", sa.Float, nullable=True),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        )

    if "session_qa" not in existing:
        op.create_table(
            "session_qa",
            sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
            sa.Column("session_id", sa.Uuid(as_uuid=True), sa.ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=False),
            sa.Column("section_order", sa.Integer, nullable=False),
            sa.Column("question_text", sa.Text, nullable=False),
            sa.Column("user_answer_text", sa.Text, nullable=True),
            sa.Column("llm_response_text", sa.Text, nullable=True),
            sa.Column("recognition_accuracy", sa.Float, nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )


def downgrade() -> None:
    op.drop_table("session_qa")
    op.drop_table("learning_sessions")
    op.drop_table("users")
