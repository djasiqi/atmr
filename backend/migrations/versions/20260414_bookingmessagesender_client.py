"""Ajout valeur CLIENT à l'enum PostgreSQL bookingmessagesender.

Revision ID: 20260414_bmsg_client
Revises: 20260412_saferpay_pay
Create Date: 2026-04-14
"""

from __future__ import annotations

from alembic import op

revision = "20260414_bmsg_client"
down_revision = "20260412_saferpay_pay"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("ALTER TYPE bookingmessagesender ADD VALUE IF NOT EXISTS 'CLIENT'")


def downgrade() -> None:
    # Retrait d'une valeur d'enum PostgreSQL non trivial — no-op.
    pass
