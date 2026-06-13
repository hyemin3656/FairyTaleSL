"""add sign_text to book_sections

Revision ID: 0008
Revises: 0007
Create Date: 2026-06-13
"""
from alembic import op
import sqlalchemy as sa

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("book_sections", sa.Column("sign_text", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("book_sections", "sign_text")
