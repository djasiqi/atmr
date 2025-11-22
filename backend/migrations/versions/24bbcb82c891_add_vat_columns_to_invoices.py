"""Add VAT columns to invoices

Revision ID: 24bbcb82c891
Revises: 8186648ac54e
Create Date: 2025-11-09 14:32:03.503163

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "24bbcb82c891"
down_revision = "8186648ac54e"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("company_billing_settings", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "vat_applicable",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            )
        )
        batch_op.add_column(
            sa.Column(
                "vat_rate",
                sa.Numeric(precision=5, scale=2),
                nullable=True,
                server_default=sa.text("7.70"),
            )
        )
        batch_op.add_column(sa.Column("vat_label", sa.String(length=50), nullable=True))
        batch_op.add_column(
            sa.Column("vat_number", sa.String(length=50), nullable=True)
        )

    with op.batch_alter_table("invoice_lines", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("vat_rate", sa.Numeric(precision=5, scale=2), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "vat_amount",
                sa.Numeric(precision=10, scale=2),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "total_with_vat",
                sa.Numeric(precision=10, scale=2),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(sa.Column("adjustment_note", sa.Text(), nullable=True))

    with op.batch_alter_table("invoices", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "vat_total_amount",
                sa.Numeric(precision=10, scale=2),
                nullable=False,
                server_default=sa.text("0"),
            )
        )
        batch_op.add_column(
            sa.Column(
                "vat_breakdown",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            )
        )

    # Backfill existing data
    op.execute(
        """
        UPDATE invoice_lines
        SET
            vat_amount = 0,
            total_with_vat = line_total
        WHERE vat_amount = 0 AND total_with_vat = 0
        """
    )
    op.execute(
        """
        UPDATE invoices
        SET vat_total_amount = 0
        WHERE vat_total_amount = 0
        """
    )

    with op.batch_alter_table("company_billing_settings", schema=None) as batch_op:
        batch_op.alter_column("vat_applicable", server_default=None)
        batch_op.alter_column("vat_rate", server_default=None)

    with op.batch_alter_table("invoice_lines", schema=None) as batch_op:
        batch_op.alter_column("vat_amount", server_default=None)
        batch_op.alter_column("total_with_vat", server_default=None)

    with op.batch_alter_table("invoices", schema=None) as batch_op:
        batch_op.alter_column("vat_total_amount", server_default=None)


def downgrade():
    with op.batch_alter_table("invoices", schema=None) as batch_op:
        batch_op.drop_column("vat_breakdown")
        batch_op.drop_column("vat_total_amount")

    with op.batch_alter_table("invoice_lines", schema=None) as batch_op:
        batch_op.drop_column("adjustment_note")
        batch_op.drop_column("total_with_vat")
        batch_op.drop_column("vat_amount")
        batch_op.drop_column("vat_rate")

    with op.batch_alter_table("company_billing_settings", schema=None) as batch_op:
        batch_op.drop_column("vat_number")
        batch_op.drop_column("vat_label")
        batch_op.drop_column("vat_rate")
        batch_op.drop_column("vat_applicable")
