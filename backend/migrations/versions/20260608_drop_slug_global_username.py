"""Drop institution slug; global case-insensitive unique username/email.

Revision ID: 20260608_global_username
Revises: 20260607_institution_identity
"""

from __future__ import annotations

from alembic import op

revision = "20260608_global_username"
down_revision = "20260607_institution_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Username indexes (replace partial scoped indexes with global lower())
    op.execute("DROP INDEX IF EXISTS ix_user_username_no_institution;")
    op.execute("DROP INDEX IF EXISTS ix_user_institution_username;")
    op.execute('DROP INDEX IF EXISTS ix_user_username;')

    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_user_username_lower
        ON public."user" (lower(username))
        WHERE username IS NOT NULL;
    """)

    # Email: drop case-sensitive unique constraint/index if present, use lower()
    op.execute('ALTER TABLE public."user" DROP CONSTRAINT IF EXISTS user_email_key;')
    op.execute('DROP INDEX IF EXISTS ix_user_email;')
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_user_email_lower
        ON public."user" (lower(email))
        WHERE email IS NOT NULL;
    """)

    # Institution slug columns (no longer used)
    op.execute("DROP INDEX IF EXISTS ix_institutions_slug;")
    op.execute("ALTER TABLE institutions DROP COLUMN IF EXISTS slug_locked;")
    op.execute("ALTER TABLE institutions DROP COLUMN IF EXISTS slug;")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_user_email_lower;")
    op.execute('CREATE UNIQUE INDEX IF NOT EXISTS ix_user_email ON public."user" (email);')

    op.execute("DROP INDEX IF EXISTS ix_user_username_lower;")
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_user_username_no_institution
        ON public."user" (username)
        WHERE institution_id IS NULL;
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_user_institution_username
        ON public."user" (institution_id, username)
        WHERE institution_id IS NOT NULL AND archived_at IS NULL;
    """)

    op.execute("""
        ALTER TABLE institutions
        ADD COLUMN IF NOT EXISTS slug VARCHAR(50) DEFAULT NULL;
    """)
    op.execute("""
        ALTER TABLE institutions
        ADD COLUMN IF NOT EXISTS slug_locked BOOLEAN NOT NULL DEFAULT true;
    """)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS ix_institutions_slug
        ON institutions (slug)
        WHERE slug IS NOT NULL;
    """)
