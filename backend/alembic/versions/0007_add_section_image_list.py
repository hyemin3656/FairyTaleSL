"""add image_urls and image_triggers to book_sections

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-27
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("book_sections", sa.Column("image_urls", postgresql.ARRAY(sa.Text()), nullable=True))
    op.add_column("book_sections", sa.Column("image_triggers", postgresql.ARRAY(sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column("book_sections", "image_triggers")
    op.drop_column("book_sections", "image_urls")
