"""add institution logo_url

Revision ID: 0d50163cb66c
Revises: 97c225be0651
Create Date: 2026-06-15 10:10:21.898181

"""

from alembic import op
import sqlalchemy as sa


revision = "0d50163cb66c"
down_revision = "97c225be0651"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "institutions",
        sa.Column("logo_url", sa.String(length=255), nullable=True),
    )


def downgrade():
    op.drop_column("institutions", "logo_url")
