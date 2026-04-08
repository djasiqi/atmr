"""Worldline Connect — paiement client réservation (hosted checkout + webhooks).

Revision ID: 20260408_worldline_checkout
Revises: 20260401_seed_plat_sub_pricing
Create Date: 2026-04-08

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260408_worldline_checkout"
down_revision = "20260401_seed_plat_sub_pricing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "payment",
        sa.Column("payment_provider", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "payment",
        sa.Column("worldline_hosted_checkout_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "payment",
        sa.Column("worldline_payment_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "payment",
        sa.Column("worldline_partial_redirect_url", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_payment_worldline_hosted_checkout_id",
        "payment",
        ["worldline_hosted_checkout_id"],
        unique=True,
    )
    op.create_index(
        "ix_payment_worldline_payment_id",
        "payment",
        ["worldline_payment_id"],
        unique=False,
    )

    op.create_table(
        "worldline_webhook_event",
        sa.Column("event_id", sa.String(length=128), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint("event_id", name="pk_worldline_webhook_event"),
    )


def downgrade() -> None:
    op.drop_table("worldline_webhook_event")
    op.drop_index("ix_payment_worldline_payment_id", table_name="payment")
    op.drop_index("ix_payment_worldline_hosted_checkout_id", table_name="payment")
    op.drop_column("payment", "worldline_partial_redirect_url")
    op.drop_column("payment", "worldline_payment_id")
    op.drop_column("payment", "worldline_hosted_checkout_id")
    op.drop_column("payment", "payment_provider")
