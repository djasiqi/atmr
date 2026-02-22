"""Add invitation fields to user table for institution user invitations.

Revision ID: invite_fields_001
Revises: None (standalone idempotent migration)
Create Date: 2026-02-09
"""

from alembic import op

# revision identifiers
revision = "invite_fields_001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Add invite_token_hash, invite_expires_at, invite_sent_at, account_status to user table."""
    # Idempotent: uses IF NOT EXISTS via raw SQL
    op.execute("""
        ALTER TABLE public."user"
        ADD COLUMN IF NOT EXISTS account_status VARCHAR(20) DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE public."user"
        ADD COLUMN IF NOT EXISTS invite_token_hash VARCHAR(64) DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE public."user"
        ADD COLUMN IF NOT EXISTS invite_expires_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE public."user"
        ADD COLUMN IF NOT EXISTS invite_sent_at TIMESTAMP WITH TIME ZONE DEFAULT NULL;
    """)
    # Index on invite_token_hash for fast token lookups
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_user_invite_token_hash
        ON public."user" (invite_token_hash)
        WHERE invite_token_hash IS NOT NULL;
    """)


def downgrade():
    """Remove invitation fields from user table."""
    op.execute("DROP INDEX IF EXISTS idx_user_invite_token_hash;")
    op.execute('ALTER TABLE public."user" DROP COLUMN IF EXISTS invite_sent_at;')
    op.execute('ALTER TABLE public."user" DROP COLUMN IF EXISTS invite_expires_at;')
    op.execute('ALTER TABLE public."user" DROP COLUMN IF EXISTS invite_token_hash;')
    op.execute('ALTER TABLE public."user" DROP COLUMN IF EXISTS account_status;')
