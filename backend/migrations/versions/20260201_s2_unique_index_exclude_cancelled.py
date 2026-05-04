"""S2 unique index: supprimer la contrainte d'unicité

Revision ID: 20260201_s2_exclude_cancelled
Revises: 20260130_material_delivery_check
Create Date: 2026-02-01

Supprime l'index unique S2. Permet plusieurs factures (annulées ou non)
pour la même clinique+période, comme pour les factures client.
"""

from alembic import op
import sqlalchemy as sa


revision = "20260201_s2_exclude_cancelled"
down_revision = "20260130_material_delivery_check"
branch_labels = None
depends_on = None


def upgrade():
    """Supprime l'index unique S2 (aucune contrainte d'unicité)."""
    connection = op.get_bind()
    result = connection.execute(
        sa.text("""
            SELECT indexname
            FROM pg_indexes
            WHERE indexname = 'uq_invoices_s2_clinic_monthly_company_clinic_period'
        """)
    ).fetchone()

    if result:
        op.drop_index(
            "uq_invoices_s2_clinic_monthly_company_clinic_period",
            table_name="invoices",
            postgresql_where=sa.text(
                "billing_strategy = 's2_clinic_monthly' AND billed_to_company_id IS NOT NULL"
            ),
        )


def downgrade():
    """Restaure l'index unique S2 (comportement d'origine)."""
    op.create_index(
        "uq_invoices_s2_clinic_monthly_company_clinic_period",
        "invoices",
        ["company_id", "billed_to_company_id", "period_year", "period_month"],
        unique=True,
        postgresql_where=sa.text(
            "billing_strategy = 's2_clinic_monthly' AND billed_to_company_id IS NOT NULL"
        ),
    )
