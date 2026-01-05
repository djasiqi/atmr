"""add_amount_paid_to_partner_invoices

Ajoute le champ amount_paid à partner_invoices pour permettre le suivi des paiements partiels.

Revision ID: add_amount_paid_partner_inv
Revises: add_exec_company_partner_inv
Create Date: 2026-01-05 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "add_amount_paid_partner_inv"
down_revision = "add_exec_company_partner_inv"  # Après add_executing_company_id_to_partner_invoice
branch_labels = None
depends_on = None


def upgrade():
    # Ajouter la colonne amount_paid si elle n'existe pas déjà
    # (peut avoir été ajoutée manuellement)
    from sqlalchemy import inspect
    from sqlalchemy.engine import reflection
    
    conn = op.get_bind()
    inspector = inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('partner_invoices')]
    
    if 'amount_paid' not in columns:
        op.add_column(
            "partner_invoices",
            sa.Column("amount_paid", sa.Numeric(10, 2), nullable=False, server_default="0.00"),
        )


def downgrade():
    # Supprimer la colonne amount_paid
    op.drop_column("partner_invoices", "amount_paid")

