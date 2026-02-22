"""Add provider column to device_tokens for FCM/Expo routing.

Revision ID: 20260221_dt_prov
Revises: 20260220_totp
Create Date: 2026-02-21
"""

from alembic import op

revision = "20260221_dt_prov"
down_revision = "20260220_totp"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE device_tokens "
        "ADD COLUMN IF NOT EXISTS provider VARCHAR(20) NOT NULL DEFAULT 'expo'"
    )


def downgrade():
    op.execute("ALTER TABLE device_tokens DROP COLUMN IF EXISTS provider")
