"""Guest checkout: booking.user_id nullable, created_via enum, payment guest FKs, unique Saferpay tx.

Revision ID: 20260422_guest_ck
Revises: 3cdd9a610dbd
Create Date: 2026-04-22
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260422_guest_ck"
down_revision = "3cdd9a610dbd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    booking_created_via = sa.Enum(
        "legacy",
        "public_guest",
        "client_app",
        "dispatcher",
        "institution_portal",
        "api_partner",
        name="booking_created_via",
    )
    booking_created_via.create(op.get_bind(), checkfirst=True)
    op.add_column(
        "booking",
        sa.Column(
            "created_via",
            booking_created_via,
            nullable=False,
            server_default="legacy",
        ),
    )
    op.alter_column("booking", "user_id", existing_type=sa.Integer(), nullable=True)

    op.alter_column("payment", "user_id", existing_type=sa.Integer(), nullable=True)
    op.alter_column("payment", "client_id", existing_type=sa.Integer(), nullable=True)

    op.drop_index("ix_payment_saferpay_transaction_id", table_name="payment")
    op.execute(
        """
        CREATE UNIQUE INDEX uq_payment_saferpay_transaction_id
        ON payment (saferpay_transaction_id)
        WHERE saferpay_transaction_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_payment_saferpay_transaction_id")
    op.create_index(
        "ix_payment_saferpay_transaction_id",
        "payment",
        ["saferpay_transaction_id"],
        unique=False,
    )

    op.alter_column("payment", "client_id", existing_type=sa.Integer(), nullable=False)
    op.alter_column("payment", "user_id", existing_type=sa.Integer(), nullable=False)

    op.alter_column("booking", "user_id", existing_type=sa.Integer(), nullable=False)
    op.drop_column("booking", "created_via")
    op.execute("DROP TYPE IF EXISTS booking_created_via")
