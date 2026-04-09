"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-08
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enum 타입 생성
    op.execute("CREATE TYPE gender_enum AS ENUM ('M', 'F', 'OTHER')")
    op.execute("CREATE TYPE role_enum AS ENUM ('child', 'guardian', 'admin')")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("username", sa.String(50), nullable=False, unique=True),
        sa.Column("nickname", sa.String(50), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("birthdate", sa.Date, nullable=True),
        sa.Column("gender", sa.Enum("M", "F", "OTHER", name="gender_enum"), nullable=True),
        sa.Column("role", sa.Enum("child", "guardian", "admin", name="role_enum"), nullable=False, server_default="child"),
        sa.Column("avatar_speed", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("subtitle_enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "books",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("cover_image_url", sa.String(500), nullable=True),
        sa.Column("target_age_min", sa.Integer, nullable=False, server_default="5"),
        sa.Column("target_age_max", sa.Integer, nullable=False, server_default="10"),
        sa.Column("categories", sa.JSON, nullable=False),
        sa.Column("author", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_books_title", "books", ["title"])

    op.create_table(
        "book_sections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("book_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False),
        sa.Column("order", sa.Integer, nullable=False),
        sa.Column("title", sa.String(200), nullable=True),
        sa.Column("text", sa.Text, nullable=False),
    )

    op.create_table(
        "gloss_motions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("gloss", sa.String(100), nullable=False, unique=True),
        sa.Column("gltf_clip_url", sa.String(500), nullable=False),
        sa.Column("emotion_label", sa.String(30), nullable=False, server_default="neutral"),
        sa.Column("blendshape_params", postgresql.JSON, nullable=True),
        sa.Column("duration_sec", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_gloss_motions_gloss", "gloss_motions", ["gloss"])

    op.create_table(
        "learning_sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("book_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("books.id", ondelete="SET NULL"), nullable=True),
        sa.Column("last_section_order", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(20), nullable=False, server_default="in_progress"),
        sa.Column("avg_recognition_accuracy", sa.Float, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "session_qa",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("learning_sessions.id", ondelete="CASCADE"), nullable=False),
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
    op.drop_index("ix_gloss_motions_gloss", table_name="gloss_motions")
    op.drop_table("gloss_motions")
    op.drop_table("book_sections")
    op.drop_index("ix_books_title", table_name="books")
    op.drop_table("books")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS gender_enum")
    op.execute("DROP TYPE IF EXISTS role_enum")
