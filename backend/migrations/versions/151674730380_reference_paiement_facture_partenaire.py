"""reference paiement facture partenaire

Revision ID: 151674730380
Revises: ec221d4c58ac
Create Date: 2026-10-27 23:09:22.406670

"""

from alembic import op
import sqlalchemy as sa


revision = "151674730380"
down_revision = "ec221d4c58ac"
branch_labels = None
depends_on = None


def upgrade():
    # Colonne détectée par autogenerate. Le reste du diff local
    # (index et colonnes hors facture partenaire) n'appartient pas à ce chantier.
    op.add_column(
        "partner_invoices",
        sa.Column("payment_reference", sa.String(length=35), nullable=True),
    )


def downgrade():
    op.drop_column("partner_invoices", "payment_reference")
