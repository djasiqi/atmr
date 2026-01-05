"""add_executing_company_id_to_partner_invoice

Ajoute le champ executing_company_id à partner_invoices et supprime
la contrainte unique sur (partnership_id, period_year, period_month)
pour permettre plusieurs factures par période si les entreprises exécutantes sont différentes.

Revision ID: add_exec_company_partner_inv
Revises: p1a2r3t4n5s7
Create Date: 2026-01-05 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "add_exec_company_partner_inv"
down_revision = "p1a2r3t4n5s7"  # Après add_partnership_status
branch_labels = None
depends_on = None


def upgrade():
    # Ajouter la colonne executing_company_id
    op.add_column(
        "partner_invoices",
        sa.Column("executing_company_id", sa.Integer(), nullable=True),
    )

    # Créer la contrainte de clé étrangère
    op.create_foreign_key(
        "fk_partner_invoices_executing_company",
        "partner_invoices",
        "company",
        ["executing_company_id"],
        ["id"],
        ondelete="CASCADE",
    )

    # Créer l'index
    op.create_index(
        "ix_partner_invoices_executing_company_id",
        "partner_invoices",
        ["executing_company_id"],
    )

    # Remplir executing_company_id pour les factures existantes
    # En utilisant le premier transfert associé pour déterminer l'entreprise exécutante
    op.execute("""
        UPDATE partner_invoices pi
        SET executing_company_id = (
            SELECT bt.executing_company_id
            FROM partner_invoice_transfers pit
            JOIN booking_transfers bt ON pit.booking_transfer_id = bt.id
            WHERE pit.partner_invoice_id = pi.id
            LIMIT 1
        )
        WHERE executing_company_id IS NULL
    """)

    # Rendre la colonne NOT NULL après avoir rempli les valeurs
    op.alter_column(
        "partner_invoices",
        "executing_company_id",
        nullable=False,
    )

    # Supprimer l'ancienne contrainte unique
    op.drop_constraint(
        "unique_partner_invoice_period",
        "partner_invoices",
        type_="unique",
    )

    # Note: On ne crée pas de nouvelle contrainte unique car on veut permettre
    # plusieurs factures pour la même période si les entreprises exécutantes sont différentes


def downgrade():
    # Recréer l'ancienne contrainte unique (sans executing_company_id)
    op.create_unique_constraint(
        "unique_partner_invoice_period",
        "partner_invoices",
        ["partnership_id", "period_year", "period_month"],
    )

    # Supprimer l'index
    op.drop_index(
        "ix_partner_invoices_executing_company_id",
        table_name="partner_invoices",
    )

    # Supprimer la contrainte de clé étrangère
    op.drop_constraint(
        "fk_partner_invoices_executing_company",
        "partner_invoices",
        type_="foreignkey",
    )

    # Supprimer la colonne executing_company_id
    op.drop_column("partner_invoices", "executing_company_id")
