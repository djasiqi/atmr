"""reference unique d'un paiement de facture

Revision ID: 152ed47ad70b
Revises: bde6118d9df4
Create Date: 2026-11-01 18:18:57.698109

Généré par Alembic sur une base estampillée ``bde6118d9df4``,
puis limité à l'index partiel de rejeu. Le reste recréait des index
déjà présents dans la chaîne et ne fait pas partie de ce lot.
"""

from alembic import op
import sqlalchemy as sa


revision = "152ed47ad70b"
down_revision = "bde6118d9df4"
branch_labels = None
depends_on = None


def upgrade():
    op.create_index(
        "uq_invoice_payments_reference",
        "invoice_payments",
        ["invoice_id", "reference"],
        unique=True,
        postgresql_where=sa.text("reference IS NOT NULL"),
    )


def downgrade():
    op.drop_index(
        "uq_invoice_payments_reference",
        table_name="invoice_payments",
        postgresql_where=sa.text("reference IS NOT NULL"),
    )
