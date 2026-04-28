"""Colonnes Saferpay sur payment (token session + id transaction).

Revision ID: 20260412_saferpay_pay
Revises: 20260410_client_type_v2
Create Date: 2026-04-12
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260412_saferpay_pay"
down_revision = "20260410_client_type_v2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "payment",
        sa.Column("saferpay_token", sa.Text(), nullable=True),
    )
    op.add_column(
        "payment",
        sa.Column("saferpay_transaction_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "payment",
        sa.Column("saferpay_notify_key", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_payment_saferpay_transaction_id",
        "payment",
        ["saferpay_transaction_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_payment_saferpay_transaction_id", table_name="payment")
    op.drop_column("payment", "saferpay_notify_key")
    op.drop_column("payment", "saferpay_transaction_id")
    op.drop_column("payment", "saferpay_token")
