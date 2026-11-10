"""conservation du CaptureId Saferpay

Revision ID: b7e4c91a02d8
Revises: c2b8f4a91e06
Create Date: 2026-10-15 00:20:00.000000

Migration manuelle limitée aux colonnes de capture sur payment.
L'autogénération de cette base propose des suppressions sans rapport.
"""

from alembic import op
import sqlalchemy as sa


revision = "b7e4c91a02d8"
down_revision = "c2b8f4a91e06"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("payment", sa.Column("saferpay_capture_id", sa.String(length=64), nullable=True))
    op.add_column(
        "payment",
        sa.Column("saferpay_captured_amount", sa.Numeric(precision=12, scale=2), nullable=True),
    )
    op.add_column(
        "payment",
        sa.Column("saferpay_captured_currency", sa.String(length=3), nullable=True),
    )
    op.add_column(
        "payment",
        sa.Column("saferpay_capture_request_id", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "payment",
        sa.Column("saferpay_capture_retry_indicator", sa.Integer(), nullable=True),
    )
    op.add_column(
        "payment",
        sa.Column("saferpay_capture_status", sa.String(length=16), nullable=True),
    )
    op.create_check_constraint(
        "ck_payment_saferpay_capture_status",
        "payment",
        "saferpay_capture_status IS NULL OR saferpay_capture_status IN "
        "('sent', 'captured', 'failed', 'uncertain', 'missing_id')",
    )
    op.create_check_constraint(
        "ck_payment_saferpay_capture_retry",
        "payment",
        "saferpay_capture_retry_indicator IS NULL OR "
        "(saferpay_capture_retry_indicator >= 0 AND saferpay_capture_retry_indicator <= 9)",
    )
    op.create_index(
        "uq_payment_saferpay_capture_id",
        "payment",
        ["saferpay_capture_id"],
        unique=True,
        postgresql_where=sa.text("saferpay_capture_id IS NOT NULL"),
    )


def downgrade():
    op.drop_index("uq_payment_saferpay_capture_id", table_name="payment")
    op.drop_constraint("ck_payment_saferpay_capture_retry", "payment", type_="check")
    op.drop_constraint("ck_payment_saferpay_capture_status", "payment", type_="check")
    op.drop_column("payment", "saferpay_capture_status")
    op.drop_column("payment", "saferpay_capture_retry_indicator")
    op.drop_column("payment", "saferpay_capture_request_id")
    op.drop_column("payment", "saferpay_captured_currency")
    op.drop_column("payment", "saferpay_captured_amount")
    op.drop_column("payment", "saferpay_capture_id")
