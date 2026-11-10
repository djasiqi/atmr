"""p7l01_credit_note_lines_corrective

Revision ID: 38c76fa54225
Revises: 9b59a05626a7
Create Date: 2026-11-02 21:00:32.251126

"""

from alembic import op
import sqlalchemy as sa


revision = "38c76fa54225"
down_revision = "9b59a05626a7"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "invoice_credit_note_line",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("credit_note_id", sa.Integer(), nullable=False),
        sa.Column("invoice_line_id", sa.Integer(), nullable=False),
        sa.Column("amount_ht", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("vat_amount", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("amount_ttc", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("vat_rate", sa.Numeric(precision=5, scale=2), nullable=True),
        sa.CheckConstraint(
            "amount_ht + vat_amount = amount_ttc",
            name="ck_invoice_credit_note_line_sum",
        ),
        sa.CheckConstraint("amount_ht >= 0", name="ck_invoice_credit_note_line_ht"),
        sa.CheckConstraint("amount_ttc > 0", name="ck_invoice_credit_note_line_ttc"),
        sa.CheckConstraint("vat_amount >= 0", name="ck_invoice_credit_note_line_vat"),
        sa.ForeignKeyConstraint(
            ["credit_note_id"],
            ["invoice_credit_note.id"],
            name="fk_invoice_credit_note_line_note",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["invoice_line_id"],
            ["invoice_lines.id"],
            name="fk_invoice_credit_note_line_invoice_line",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "credit_note_id",
            "invoice_line_id",
            name="uq_invoice_credit_note_line_note_line",
        ),
    )
    op.create_index(
        "ix_invoice_credit_note_line_invoice_line",
        "invoice_credit_note_line",
        ["invoice_line_id"],
        unique=False,
    )

    with op.batch_alter_table("invoices", schema=None) as batch_op:
        batch_op.add_column(sa.Column("corrects_invoice_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("correction_credit_note_id", sa.Integer(), nullable=True)
        )
        batch_op.add_column(sa.Column("correction_reason", sa.Text(), nullable=True))
        batch_op.add_column(
            sa.Column("correction_idempotency_key", sa.String(length=80), nullable=True)
        )
        batch_op.create_index(
            batch_op.f("ix_invoices_correction_credit_note_id"),
            ["correction_credit_note_id"],
            unique=False,
        )
        batch_op.create_index(
            batch_op.f("ix_invoices_corrects_invoice_id"),
            ["corrects_invoice_id"],
            unique=False,
        )
        batch_op.create_unique_constraint(
            "uq_invoice_correction_idempotency",
            ["company_id", "correction_idempotency_key"],
        )
        batch_op.create_foreign_key(
            "fk_invoices_corrects_invoice_id",
            "invoices",
            ["corrects_invoice_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_foreign_key(
            "fk_invoices_correction_credit_note_id",
            "invoice_credit_note",
            ["correction_credit_note_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade():
    with op.batch_alter_table("invoices", schema=None) as batch_op:
        batch_op.drop_constraint("fk_invoices_correction_credit_note_id", type_="foreignkey")
        batch_op.drop_constraint("fk_invoices_corrects_invoice_id", type_="foreignkey")
        batch_op.drop_constraint("uq_invoice_correction_idempotency", type_="unique")
        batch_op.drop_index(batch_op.f("ix_invoices_corrects_invoice_id"))
        batch_op.drop_index(batch_op.f("ix_invoices_correction_credit_note_id"))
        batch_op.drop_column("correction_idempotency_key")
        batch_op.drop_column("correction_reason")
        batch_op.drop_column("correction_credit_note_id")
        batch_op.drop_column("corrects_invoice_id")

    op.drop_index(
        "ix_invoice_credit_note_line_invoice_line",
        table_name="invoice_credit_note_line",
    )
    op.drop_table("invoice_credit_note_line")
