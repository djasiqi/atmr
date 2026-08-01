"""booking_invoice_institution_patient_id

Revision ID: 801a7e0a2923
Revises: c8f1a2b3d4e5
Create Date: 2026-08-01 00:12:06.270033

"""

from alembic import op
import sqlalchemy as sa


revision = "801a7e0a2923"
down_revision = "c8f1a2b3d4e5"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "booking",
        sa.Column("institution_patient_id", sa.Integer(), nullable=True),
    )
    op.create_index(
        op.f("ix_booking_institution_patient_id"),
        "booking",
        ["institution_patient_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_booking_institution_patient_id",
        "booking",
        "institution_patients",
        ["institution_patient_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.add_column(
        "invoices",
        sa.Column("institution_patient_id", sa.Integer(), nullable=True),
    )
    op.create_index(
        op.f("ix_invoices_institution_patient_id"),
        "invoices",
        ["institution_patient_id"],
        unique=False,
    )
    op.create_index(
        "ix_invoice_company_institution_patient_period",
        "invoices",
        [
            "company_id",
            "institution_patient_id",
            "period_year",
            "period_month",
        ],
        unique=False,
    )
    op.create_foreign_key(
        "fk_invoices_institution_patient_id",
        "invoices",
        "institution_patients",
        ["institution_patient_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    op.drop_constraint(
        "fk_invoices_institution_patient_id", "invoices", type_="foreignkey"
    )
    op.drop_index(
        "ix_invoice_company_institution_patient_period", table_name="invoices"
    )
    op.drop_index(op.f("ix_invoices_institution_patient_id"), table_name="invoices")
    op.drop_column("invoices", "institution_patient_id")

    op.drop_constraint(
        "fk_booking_institution_patient_id", "booking", type_="foreignkey"
    )
    op.drop_index(op.f("ix_booking_institution_patient_id"), table_name="booking")
    op.drop_column("booking", "institution_patient_id")
