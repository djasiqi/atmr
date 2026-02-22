"""Add TOTP 2FA fields to user and security_policy to company.

Revision ID: 20260220_totp
Revises: 20260220_audit_res
Create Date: 2026-02-20
"""

from alembic import op

revision = "20260220_totp"
down_revision = "20260220_audit_res"
branch_labels = None
depends_on = None


def upgrade():
    op.execute('ALTER TABLE "user" ADD COLUMN IF NOT EXISTS totp_secret_encrypted TEXT')
    op.execute('ALTER TABLE "user" ADD COLUMN IF NOT EXISTS totp_enabled BOOLEAN DEFAULT FALSE')
    op.execute('ALTER TABLE "user" ADD COLUMN IF NOT EXISTS totp_enabled_at TIMESTAMP WITH TIME ZONE')
    op.execute('ALTER TABLE "user" ADD COLUMN IF NOT EXISTS recovery_codes_hash TEXT')
    op.execute('ALTER TABLE "user" ADD COLUMN IF NOT EXISTS recovery_codes_remaining INTEGER DEFAULT 0')
    op.execute("ALTER TABLE company ADD COLUMN IF NOT EXISTS security_policy TEXT")


def downgrade():
    op.execute("ALTER TABLE company DROP COLUMN IF EXISTS security_policy")
    op.execute('ALTER TABLE "user" DROP COLUMN IF EXISTS recovery_codes_remaining')
    op.execute('ALTER TABLE "user" DROP COLUMN IF EXISTS recovery_codes_hash')
    op.execute('ALTER TABLE "user" DROP COLUMN IF EXISTS totp_enabled_at')
    op.execute('ALTER TABLE "user" DROP COLUMN IF EXISTS totp_enabled')
    op.execute('ALTER TABLE "user" DROP COLUMN IF EXISTS totp_secret_encrypted')
