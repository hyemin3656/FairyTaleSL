"""add image_url to book_sections

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-19
"""
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "book_sections",
        sa.Column("image_url", sa.String(500), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("book_sections", "image_url")
