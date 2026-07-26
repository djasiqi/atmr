"""Institution identity management: slugs, user auth fields, audit, reserved usernames.

Revision ID: 20260607_institution_identity
Revises: 20260603_manual_invariant
"""

from __future__ import annotations

import re

import sqlalchemy as sa
from alembic import op

revision = "20260607_institution_identity"
down_revision = "20260603_manual_invariant"
branch_labels = None
depends_on = None


def _slugify(name: str) -> str:
    text = (name or "").strip().lower()
    text = text.replace("'", "-").replace("'", "-")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:50] or "institution"


def upgrade() -> None:
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

    conn = op.get_bind()
    rows = conn.execute(
        sa.text("SELECT id, name FROM institutions WHERE slug IS NULL OR slug = ''")
    ).fetchall()

    used_slugs: set[str] = set()
    existing = conn.execute(
        sa.text("SELECT slug FROM institutions WHERE slug IS NOT NULL")
    ).fetchall()
    for (slug,) in existing:
        if slug:
            used_slugs.add(str(slug))

    for inst_id, name in rows:
        base = _slugify(str(name))
        candidate = base
        suffix = 2
        while candidate in used_slugs:
            candidate = f"{base}-{suffix}"[:50]
            suffix += 1
        used_slugs.add(candidate)
        conn.execute(
            sa.text(
                "UPDATE institutions SET slug = :slug, slug_locked = true WHERE id = :id"
            ),
            {"slug": candidate, "id": inst_id},
        )

    op.execute(
        "ALTER TABLE public.\"user\" ADD COLUMN IF NOT EXISTS authentication_method VARCHAR(20) DEFAULT 'email';"
    )
    op.execute(
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS temporary_password_created_at TIMESTAMPTZ DEFAULT NULL;'
    )
    op.execute(
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS last_password_reset_at TIMESTAMPTZ DEFAULT NULL;'
    )
    op.execute(
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS temp_password_generation_count INTEGER NOT NULL DEFAULT 0;'
    )
    op.execute(
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS first_login_completed_at TIMESTAMPTZ DEFAULT NULL;'
    )
    op.execute(
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS disabled_at TIMESTAMPTZ DEFAULT NULL;'
    )
    op.execute(
        'ALTER TABLE public."user" ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ DEFAULT NULL;'
    )

    op.execute("DROP INDEX IF EXISTS ix_user_username;")
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
        CREATE TABLE IF NOT EXISTS institution_user_audit_events (
            id SERIAL PRIMARY KEY,
            institution_id INTEGER NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
            target_user_id INTEGER NOT NULL REFERENCES public."user"(id) ON DELETE CASCADE,
            performed_by_user_id INTEGER REFERENCES public."user"(id) ON DELETE SET NULL,
            event_type VARCHAR(50) NOT NULL,
            performed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ip_address VARCHAR(45),
            user_agent TEXT,
            metadata JSONB
        );
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_institution_user_audit_institution
        ON institution_user_audit_events (institution_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_institution_user_audit_target
        ON institution_user_audit_events (target_user_id);
    """)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_institution_user_audit_event_type
        ON institution_user_audit_events (event_type);
    """)

    op.execute("""
        CREATE TABLE IF NOT EXISTS institution_reserved_usernames (
            id SERIAL PRIMARY KEY,
            institution_id INTEGER NOT NULL REFERENCES institutions(id) ON DELETE CASCADE,
            username VARCHAR(100) NOT NULL,
            reserved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            reserved_reason VARCHAR(50) NOT NULL DEFAULT 'user_archived',
            former_user_id INTEGER,
            CONSTRAINT uq_institution_reserved_username UNIQUE (institution_id, username)
        );
    """)


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS institution_reserved_usernames;")
    op.execute("DROP TABLE IF EXISTS institution_user_audit_events;")
    op.execute("DROP INDEX IF EXISTS ix_user_institution_username;")
    op.execute("DROP INDEX IF EXISTS ix_user_username_no_institution;")
    op.execute(
        'CREATE UNIQUE INDEX IF NOT EXISTS ix_user_username ON public."user" (username);'
    )
    op.execute('ALTER TABLE public."user" DROP COLUMN IF EXISTS archived_at;')
    op.execute('ALTER TABLE public."user" DROP COLUMN IF EXISTS disabled_at;')
    op.execute(
        'ALTER TABLE public."user" DROP COLUMN IF EXISTS first_login_completed_at;'
    )
    op.execute(
        'ALTER TABLE public."user" DROP COLUMN IF EXISTS temp_password_generation_count;'
    )
    op.execute(
        'ALTER TABLE public."user" DROP COLUMN IF EXISTS last_password_reset_at;'
    )
    op.execute(
        'ALTER TABLE public."user" DROP COLUMN IF EXISTS temporary_password_created_at;'
    )
    op.execute('ALTER TABLE public."user" DROP COLUMN IF EXISTS authentication_method;')
    op.execute("DROP INDEX IF EXISTS ix_institutions_slug;")
    op.execute("ALTER TABLE institutions DROP COLUMN IF EXISTS slug_locked;")
    op.execute("ALTER TABLE institutions DROP COLUMN IF EXISTS slug;")
