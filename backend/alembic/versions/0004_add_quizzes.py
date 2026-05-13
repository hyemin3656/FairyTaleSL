"""quizzes 테이블 추가

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-13
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "quizzes",
        sa.Column("id", sa.Uuid(as_uuid=True), primary_key=True),
        sa.Column(
            "book_id", sa.Uuid(as_uuid=True),
            sa.ForeignKey("books.id", ondelete="CASCADE"), nullable=False, index=True,
        ),
        sa.Column("section_order", sa.Integer, nullable=False),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("expected_answer", sa.String(200), nullable=False),
        sa.Column("synonyms", sa.JSON, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("book_id", "section_order", name="uq_quiz_book_section"),
    )


def downgrade() -> None:
    op.drop_table("quizzes")
