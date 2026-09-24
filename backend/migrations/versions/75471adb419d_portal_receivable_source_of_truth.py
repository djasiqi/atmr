"""portal receivable source of truth

Revision ID: 75471adb419d
Revises: 02c0836f2261
Create Date: 2026-09-23 22:53:42.180442

Tables générées par ``flask db revision --autogenerate``.
Les écarts d'index / colonnes sans rapport ont été retirés.
Pas de ``batch_alter_table`` : création de tables neuves uniquement.
"""

import sqlalchemy as sa
from alembic import op

revision = "75471adb419d"
down_revision = "02c0836f2261"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "portal_receivable",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("creditor_company_id", sa.Integer(), nullable=False),
        sa.Column("creditor_name_snapshot", sa.String(length=200), nullable=False),
        sa.Column("debtor_user_id", sa.Integer(), nullable=False),
        sa.Column("debtor_name_snapshot", sa.String(length=200), nullable=False),
        sa.Column("debtor_email_snapshot", sa.String(length=255), nullable=True),
        sa.Column("debtor_phone_snapshot", sa.String(length=40), nullable=True),
        sa.Column("debtor_billing_address_snapshot", sa.Text(), nullable=True),
        sa.Column("external_invoice_number", sa.String(length=80), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("due_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("total_amount", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("amount_paid", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("balance_due", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("disputed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dispute_reason", sa.Text(), nullable=True),
        sa.Column("disputed_by_user_id", sa.Integer(), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_by_user_id", sa.Integer(), nullable=True),
        sa.Column("cancellation_reason", sa.Text(), nullable=True),
        sa.Column("recorded_by_user_id", sa.Integer(), nullable=False),
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
            "status IN ("
            "'issued', 'partially_paid', 'paid', 'overdue', 'disputed', 'cancelled'"
            ")",
            name="ck_portal_receivable_status",
        ),
        sa.CheckConstraint("amount_paid >= 0", name="ck_portal_receivable_paid_nonneg"),
        sa.CheckConstraint(
            "balance_due >= 0", name="ck_portal_receivable_balance_nonneg"
        ),
        sa.CheckConstraint(
            "total_amount >= 0", name="ck_portal_receivable_total_nonneg"
        ),
        sa.ForeignKeyConstraint(
            ["cancelled_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["creditor_company_id"], ["company.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["debtor_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["disputed_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["recorded_by_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "creditor_company_id",
            "external_invoice_number",
            name="uq_portal_receivable_creditor_invoice",
        ),
    )
    op.create_index(
        "ix_portal_receivable_creditor_company_id",
        "portal_receivable",
        ["creditor_company_id"],
        unique=False,
    )
    op.create_index(
        "ix_portal_receivable_debtor",
        "portal_receivable",
        ["debtor_user_id"],
        unique=False,
    )
    op.create_index(
        "ix_portal_receivable_due_date",
        "portal_receivable",
        ["due_date"],
        unique=False,
    )
    op.create_index(
        "ix_portal_receivable_status",
        "portal_receivable",
        ["status"],
        unique=False,
    )

    op.create_table(
        "portal_receivable_line",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("receivable_id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("booking_contract_event_id", sa.Integer(), nullable=False),
        sa.Column("invoiced_amount", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.CheckConstraint(
            "invoiced_amount >= 0", name="ck_portal_receivable_line_amount_nonneg"
        ),
        sa.ForeignKeyConstraint(
            ["booking_contract_event_id"],
            ["client_booking_contract_event.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["receivable_id"], ["portal_receivable.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portal_receivable_line_booking",
        "portal_receivable_line",
        ["booking_id"],
        unique=False,
    )
    op.create_index(
        "ix_portal_receivable_line_receivable_id",
        "portal_receivable_line",
        ["receivable_id"],
        unique=False,
    )

    op.create_table(
        "portal_receivable_payment",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("receivable_id", sa.Integer(), nullable=False),
        sa.Column("amount", sa.Numeric(precision=12, scale=2), nullable=False),
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=False),
        sa.Column("reference", sa.String(length=120), nullable=True),
        sa.Column("recorded_by_user_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "method IN ('bank_transfer', 'cash', 'card', 'other')",
            name="ck_portal_receivable_payment_method",
        ),
        sa.CheckConstraint(
            "amount > 0", name="ck_portal_receivable_payment_amount_positive"
        ),
        sa.ForeignKeyConstraint(
            ["receivable_id"], ["portal_receivable.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["recorded_by_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portal_receivable_payment_receivable_id",
        "portal_receivable_payment",
        ["receivable_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_portal_receivable_payment_receivable_id",
        table_name="portal_receivable_payment",
    )
    op.drop_table("portal_receivable_payment")
    op.drop_index(
        "ix_portal_receivable_line_receivable_id",
        table_name="portal_receivable_line",
    )
    op.drop_index(
        "ix_portal_receivable_line_booking",
        table_name="portal_receivable_line",
    )
    op.drop_table("portal_receivable_line")
    op.drop_index("ix_portal_receivable_status", table_name="portal_receivable")
    op.drop_index("ix_portal_receivable_due_date", table_name="portal_receivable")
    op.drop_index("ix_portal_receivable_debtor", table_name="portal_receivable")
    op.drop_index(
        "ix_portal_receivable_creditor_company_id",
        table_name="portal_receivable",
    )
    op.drop_table("portal_receivable")
