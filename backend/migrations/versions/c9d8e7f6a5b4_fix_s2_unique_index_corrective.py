"""fix_s2_unique_index_corrective

Revision ID: c9d8e7f6a5b4
Revises: 5e9c90875469
Create Date: 2026-01-21 22:00:00.000000

Migration corrective pour recréer l'index unique S2 qui a été supprimé par erreur
dans la migration 5e9c90875469_add_billing_source_and_transport_.

Cette migration vérifie si l'index existe, et le recrée s'il est absent.
Idempotent : peut être exécutée plusieurs fois sans erreur.
"""
from alembic import op
import sqlalchemy as sa


revision = "c9d8e7f6a5b4"
down_revision = "5e9c90875469"
branch_labels = None
depends_on = None


def upgrade():
    """Recrée l'index unique S2 s'il n'existe pas."""
    # Vérifier si l'index existe déjà
    connection = op.get_bind()
    result = connection.execute(
        sa.text("""
            SELECT indexname
            FROM pg_indexes
            WHERE indexname = 'uq_invoices_s2_clinic_monthly_company_clinic_period'
        """)
    ).fetchone()

    if not result:
        # L'index n'existe pas, le recréer
        op.create_index(
            "uq_invoices_s2_clinic_monthly_company_clinic_period",
            "invoices",
            ["company_id", "billed_to_company_id", "period_year", "period_month"],
            unique=True,
            postgresql_where=sa.text(
                "billing_strategy = 's2_clinic_monthly' AND billed_to_company_id IS NOT NULL"
            ),
        )


def downgrade():
    """Supprime l'index unique S2 (pour rollback si nécessaire)."""
    # Vérifier si l'index existe avant de le supprimer
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
