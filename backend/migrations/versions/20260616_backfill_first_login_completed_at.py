"""Backfill first_login_completed_at pour comptes institution username déjà activés.

Revision ID: 20260616_backfill_first_login
Revises: 20260611_institution_timeline
"""

from __future__ import annotations

from alembic import op

revision = "20260616_backfill_first_login"
down_revision = "20260611_institution_timeline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        UPDATE public."user"
        SET first_login_completed_at = COALESCE(updated_at, created_at, NOW())
        WHERE institution_id IS NOT NULL
          AND authentication_method = 'username'
          AND first_login_completed_at IS NULL
          AND force_password_change IS FALSE
          AND archived_at IS NULL;
    """)


def downgrade() -> None:
    pass
