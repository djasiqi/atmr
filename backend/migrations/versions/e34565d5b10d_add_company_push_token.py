"""add_company_push_token

Revision ID: e34565d5b10d
Revises: 81ffc072d63d
Create Date: 2026-01-20 16:18:59.472777

"""

from alembic import op

revision = "e34565d5b10d"
down_revision = "81ffc072d63d"
branch_labels = None
depends_on = None


def upgrade():
    # Idempotent: skip if column already exists (e.g. applied manually)
    op.execute('ALTER TABLE "user" ADD COLUMN IF NOT EXISTS push_token VARCHAR(255)')
    op.execute('CREATE INDEX IF NOT EXISTS ix_user_push_token ON "user" (push_token)')


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_user_push_token")
    op.execute('ALTER TABLE "user" DROP COLUMN IF EXISTS push_token')
