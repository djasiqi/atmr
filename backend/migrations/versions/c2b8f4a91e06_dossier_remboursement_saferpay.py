"""dossier de remboursement saferpay

Revision ID: c2b8f4a91e06
Revises: 9ddf4ec5763f
Create Date: 2026-10-14 23:45:00.000000

Migration manuelle limitée aux tables du dossier. L'autogénération
précédente proposait des suppressions sans rapport et a été écartée.
"""

from alembic import op
import sqlalchemy as sa


revision = "c2b8f4a91e06"
down_revision = "9ddf4ec5763f"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "saferpay_refund_case",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("payment_id", sa.Integer(), nullable=False),
        sa.Column("holder_user_id", sa.Integer(), nullable=False),
        sa.Column("holder_public_id", sa.String(length=64), nullable=False),
        sa.Column("holder_label", sa.String(length=255), nullable=False),
        sa.Column("saferpay_transaction_id", sa.String(length=128), nullable=False),
        sa.Column("captured_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("prior_confirmed_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("remaining_at_open", sa.Numeric(12, 2), nullable=False),
        sa.Column("proposed_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("eligibility_justification", sa.Text(), nullable=False),
        sa.Column("request_status", sa.String(length=16), nullable=False),
        sa.Column("approver_user_id", sa.Integer(), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("decision_note", sa.Text(), nullable=True),
        sa.Column("execution_status", sa.String(length=16), nullable=False),
        sa.Column("operation_reference", sa.String(length=64), nullable=True),
        sa.Column("provider_refund_reference", sa.String(length=128), nullable=True),
        sa.Column("confirmed_amount", sa.Numeric(12, 2), nullable=True),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("proof_text", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "request_status IN ('created', 'approved', 'refused')",
            name="ck_saferpay_refund_request_status",
        ),
        sa.CheckConstraint(
            "execution_status IN ('not_sent', 'sent', 'confirmed', 'failed', 'uncertain')",
            name="ck_saferpay_refund_execution_status",
        ),
        sa.CheckConstraint("currency = 'CHF'", name="ck_saferpay_refund_currency_chf"),
        sa.CheckConstraint(
            "captured_amount > 0", name="ck_saferpay_refund_captured_positive"
        ),
        sa.CheckConstraint(
            "proposed_amount > 0", name="ck_saferpay_refund_proposed_positive"
        ),
        sa.CheckConstraint(
            "proposed_amount <= captured_amount",
            name="ck_saferpay_refund_proposed_within_capture",
        ),
        sa.CheckConstraint(
            "remaining_at_open >= 0", name="ck_saferpay_refund_remaining_nonneg"
        ),
        sa.CheckConstraint(
            "prior_confirmed_amount >= 0", name="ck_saferpay_refund_prior_nonneg"
        ),
        sa.CheckConstraint(
            "proposed_amount <= remaining_at_open",
            name="ck_saferpay_refund_proposed_within_remaining",
        ),
        sa.CheckConstraint(
            "confirmed_amount IS NULL OR "
            "(confirmed_amount > 0 AND confirmed_amount <= proposed_amount)",
            name="ck_saferpay_refund_confirmed_within_proposed",
        ),
        sa.CheckConstraint(
            "(request_status = 'created' AND approver_user_id IS NULL) OR "
            "(request_status IN ('approved', 'refused') AND approver_user_id IS NOT NULL)",
            name="ck_saferpay_refund_decision_approver",
        ),
        sa.CheckConstraint(
            "execution_status <> 'confirmed' OR ("
            "provider_refund_reference IS NOT NULL AND confirmed_amount IS NOT NULL)",
            name="ck_saferpay_refund_confirmed_has_proof",
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["payment_id"], ["payment.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["holder_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["approver_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
        sa.UniqueConstraint("operation_reference"),
    )
    op.create_index(
        "ix_saferpay_refund_case_payment",
        "saferpay_refund_case",
        ["payment_id"],
    )
    op.create_index(
        "ix_saferpay_refund_case_booking",
        "saferpay_refund_case",
        ["booking_id"],
    )
    op.create_index(
        "uq_saferpay_refund_provider_reference",
        "saferpay_refund_case",
        ["provider_refund_reference"],
        unique=True,
        postgresql_where=sa.text("provider_refund_reference IS NOT NULL"),
    )
    op.create_table(
        "saferpay_refund_case_event",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "payload_text",
            sa.Text(),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["case_id"], ["saferpay_refund_case.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_saferpay_refund_event_case",
        "saferpay_refund_case_event",
        ["case_id"],
    )


def downgrade():
    op.drop_index(
        "ix_saferpay_refund_event_case", table_name="saferpay_refund_case_event"
    )
    op.drop_table("saferpay_refund_case_event")
    op.drop_index(
        "uq_saferpay_refund_provider_reference",
        table_name="saferpay_refund_case",
        postgresql_where=sa.text("provider_refund_reference IS NOT NULL"),
    )
    op.drop_index("ix_saferpay_refund_case_booking", table_name="saferpay_refund_case")
    op.drop_index("ix_saferpay_refund_case_payment", table_name="saferpay_refund_case")
    op.drop_table("saferpay_refund_case")
