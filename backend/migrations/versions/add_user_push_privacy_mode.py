"""add_user_push_privacy_mode

Revision ID: add_user_push_privacy
Revises: a7b8c9d0e1f2, add_sent_at_partner_inv
Create Date: 2026-01-27

Mode discret push : "detailed" | "discreet" (pas de nom client sur lockscreen).
"""

from alembic import op

revision = "add_user_push_privacy"
down_revision = ("a7b8c9d0e1f2", "add_sent_at_partner_inv")
branch_labels = None
depends_on = None


def upgrade():
    # push_token peut manquer si cette branche n’a pas passé par add_company_push_token
    op.execute('ALTER TABLE "user" ADD COLUMN IF NOT EXISTS push_token VARCHAR(255)')
    op.execute('CREATE INDEX IF NOT EXISTS ix_user_push_token ON "user" (push_token)')
    op.execute(
        'ALTER TABLE "user" ADD COLUMN IF NOT EXISTS push_privacy_mode VARCHAR(20)'
    )


def downgrade():
    op.execute('ALTER TABLE "user" DROP COLUMN IF EXISTS push_privacy_mode')
    op.execute("DROP INDEX IF EXISTS ix_user_push_token")
    op.execute('ALTER TABLE "user" DROP COLUMN IF EXISTS push_token')
