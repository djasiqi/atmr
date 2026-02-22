"""Add cancellation_policy JSONB to company_billing_settings and fee columns to booking.

Revision ID: 20260218_cancel_policy
Revises: 20260218_dedupe
Create Date: 2026-02-18
"""

from alembic import op

revision = "20260218_cancel_policy"
down_revision = "20260218_dedupe"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE company_billing_settings "
        "ADD COLUMN IF NOT EXISTS cancellation_policy JSONB"
    )
    op.execute(
        "ALTER TABLE booking "
        "ADD COLUMN IF NOT EXISTS cancellation_fee_amount NUMERIC(10,2)"
    )
    op.execute(
        "ALTER TABLE booking "
        "ADD COLUMN IF NOT EXISTS cancellation_fee_percent INTEGER"
    )
    op.execute(
        "ALTER TABLE booking "
        "ADD COLUMN IF NOT EXISTS cancellation_fee_tier_id VARCHAR(50)"
    )


def downgrade():
    op.execute("ALTER TABLE booking DROP COLUMN IF EXISTS cancellation_fee_tier_id")
    op.execute("ALTER TABLE booking DROP COLUMN IF EXISTS cancellation_fee_percent")
    op.execute("ALTER TABLE booking DROP COLUMN IF EXISTS cancellation_fee_amount")
    op.execute(
        "ALTER TABLE company_billing_settings "
        "DROP COLUMN IF EXISTS cancellation_policy"
    )
