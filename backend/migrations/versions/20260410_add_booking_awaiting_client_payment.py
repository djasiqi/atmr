"""Ajoute le statut booking AWAITING_CLIENT_PAYMENT (paiement avant dispatch).

Revision ID: 20260410_await_pay
Revises: 20260409_activation_session
Create Date: 2026-04-10
"""

from __future__ import annotations

from alembic import op

revision = "20260410_await_pay"
down_revision = "20260409_activation_session"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TYPE booking_status ADD VALUE IF NOT EXISTS 'AWAITING_CLIENT_PAYMENT'"
    )


def downgrade() -> None:
    # PostgreSQL ne permet pas de retirer une valeur d'enum proprement sans recréer le type.
    pass
