"""Index booking(created_at) pour agrégations admin / tendances mensuelles.

Revision ID: 20260326_ix_booking_created_at
Revises: 20260306_demo_accesses
Create Date: 2026-03-26

"""

from alembic import op
from sqlalchemy import text

revision = "20260326_ix_booking_created_at"
down_revision = "20260306_demo_accesses"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        text(
            "CREATE INDEX IF NOT EXISTS ix_booking_created_at "
            "ON booking (created_at)"
        )
    )


def downgrade() -> None:
    op.execute(text("DROP INDEX IF EXISTS ix_booking_created_at"))
