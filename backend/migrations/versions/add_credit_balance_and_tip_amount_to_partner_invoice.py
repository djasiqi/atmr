"""add_credit_balance_and_tip_amount_to_partner_invoice

Ajoute les champs credit_balance et tip_amount à partner_invoices
pour gérer les paiements excédentaires (crédit à déduire ou pourboire).

Revision ID: add_credit_tip_partner_inv
Revises: add_exec_company_partner_inv
Create Date: 2026-01-05 13:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "add_credit_tip_partner_inv"
down_revision = (
    "add_amount_paid_partner_inv"  # Après add_amount_paid_to_partner_invoices
)
branch_labels = None
depends_on = None


def upgrade():
    """
    Ajoute les champs credit_balance et tip_amount à partner_invoices
    pour gérer les paiements excédentaires (crédit à déduire ou pourboire).

    ⚠️ NOTE: Cette migration est idempotente - elle vérifie l'existence
    des colonnes avant de les ajouter. En production, les colonnes peuvent
    avoir été créées manuellement avant l'application de cette migration.
    """
    # Vérifier l'existence des colonnes avant de les ajouter
    from sqlalchemy import inspect
    from sqlalchemy.engine import reflection

    bind = op.get_bind()
    inspector = reflection.Inspector.from_engine(bind)
    existing_columns = [
        col["name"] for col in inspector.get_columns("partner_invoices")
    ]

    # Ajouter la colonne credit_balance (si elle n'existe pas déjà)
    if "credit_balance" not in existing_columns:
        op.add_column(
            "partner_invoices",
            sa.Column(
                "credit_balance",
                sa.Numeric(10, 2),
                nullable=False,
                server_default="0.00",
            ),
        )

    # Ajouter la colonne tip_amount (si elle n'existe pas déjà)
    if "tip_amount" not in existing_columns:
        op.add_column(
            "partner_invoices",
            sa.Column(
                "tip_amount", sa.Numeric(10, 2), nullable=False, server_default="0.00"
            ),
        )


def downgrade():
    # Supprimer la colonne tip_amount
    op.drop_column("partner_invoices", "tip_amount")

    # Supprimer la colonne credit_balance
    op.drop_column("partner_invoices", "credit_balance")
