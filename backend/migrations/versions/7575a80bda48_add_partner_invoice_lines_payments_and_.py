"""add partner invoice lines payments and recipient overrides

Revision ID: 7575a80bda48
Revises: 3d4ba7a5e8ab
Create Date: 2026-09-17 08:54:14.771622

"""

from alembic import op
import sqlalchemy as sa


revision = "7575a80bda48"
down_revision = "3d4ba7a5e8ab"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "partner_invoice_lines",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("partner_invoice_id", sa.Integer(), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=False),
        sa.Column("quantity", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("unit_price", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("amount", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("vat_rate", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=True),
        sa.Column("sort_order", sa.Integer(), nullable=False),
        sa.Column("service_date", sa.String(length=20), nullable=True),
        sa.Column("client_name", sa.String(length=200), nullable=True),
        sa.Column("departure", sa.String(length=500), nullable=True),
        sa.Column("arrival", sa.String(length=500), nullable=True),
        sa.Column("note", sa.String(length=500), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["partner_invoice_id"], ["partner_invoices.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_partner_invoice_lines_partner_invoice_id"),
        "partner_invoice_lines",
        ["partner_invoice_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_partner_invoice_lines_source_id"),
        "partner_invoice_lines",
        ["source_id"],
        unique=False,
    )
    op.create_table(
        "partner_invoice_payments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("partner_invoice_id", sa.Integer(), nullable=False),
        sa.Column("amount", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("method", sa.String(length=50), nullable=False),
        sa.Column(
            "paid_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["partner_invoice_id"], ["partner_invoices.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_partner_invoice_payments_partner_invoice_id"),
        "partner_invoice_payments",
        ["partner_invoice_id"],
        unique=False,
    )
    op.add_column(
        "partner_invoices",
        sa.Column("recipient_name", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "partner_invoices",
        sa.Column("recipient_address", sa.String(length=1000), nullable=True),
    )
    op.add_column(
        "partner_invoices",
        sa.Column("recipient_contact", sa.String(length=300), nullable=True),
    )


def downgrade():
    op.drop_column("partner_invoices", "recipient_contact")
    op.drop_column("partner_invoices", "recipient_address")
    op.drop_column("partner_invoices", "recipient_name")
    op.drop_index(
        op.f("ix_partner_invoice_payments_partner_invoice_id"),
        table_name="partner_invoice_payments",
    )
    op.drop_table("partner_invoice_payments")
    op.drop_index(
        op.f("ix_partner_invoice_lines_source_id"),
        table_name="partner_invoice_lines",
    )
    op.drop_index(
        op.f("ix_partner_invoice_lines_partner_invoice_id"),
        table_name="partner_invoice_lines",
    )
    op.drop_table("partner_invoice_lines")
