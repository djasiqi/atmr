"""Add soft rotation fields to refresh_token.

Revision ID: 20260221_soft_rot
Revises: 20260220_totp
Create Date: 2026-02-21
"""

from alembic import op

revision = "20260221_soft_rot"
down_revision = "20260220_totp"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE refresh_token ADD COLUMN IF NOT EXISTS rotated_to_hash VARCHAR(64)"
    )
    op.execute(
        "ALTER TABLE refresh_token ADD COLUMN IF NOT EXISTS rotated_at TIMESTAMP WITH TIME ZONE"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_refresh_token_rotated_to_hash "
        "ON refresh_token (rotated_to_hash)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_refresh_token_rotated_to_hash")
    op.execute("ALTER TABLE refresh_token DROP COLUMN IF EXISTS rotated_at")
    op.execute("ALTER TABLE refresh_token DROP COLUMN IF EXISTS rotated_to_hash")
