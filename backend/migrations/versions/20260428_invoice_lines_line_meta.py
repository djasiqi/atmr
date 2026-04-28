"""invoice_lines: line_meta JSONB (prestation perso mode temps/quantité pour PDF)

Revision ID: 20260428_line_meta
Revises: 20260427_bt_owner_status
Create Date: 2026-04-28

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "20260428_line_meta"
down_revision = "20260427_bt_owner_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("invoice_lines", sa.Column("line_meta", JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column("invoice_lines", "line_meta")
