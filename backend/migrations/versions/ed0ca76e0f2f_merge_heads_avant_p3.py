
"""merge heads avant P3

Revision ID: ed0ca76e0f2f
Revises: ('0d50163cb66c', '20260616_backfill_first_login')
Create Date: 2026-06-16 16:44:17.532777

"""
from alembic import op
import sqlalchemy as sa


revision = "ed0ca76e0f2f"
down_revision = ("0d50163cb66c", "20260616_backfill_first_login")
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
