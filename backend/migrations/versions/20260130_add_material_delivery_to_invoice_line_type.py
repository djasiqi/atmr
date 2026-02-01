"""add material_delivery to invoice_line_type enum

Revision ID: 20260130_invoice_line_type
Revises: 20260130_material_delivery_price
Create Date: 2026-01-30

"""

from alembic import op


revision = "20260130_invoice_line_type"
down_revision = "20260130_material_delivery_price"
branch_labels = None
depends_on = None


def upgrade():
    # PostgreSQL: ajouter la valeur material_delivery à l'enum invoice_line_type
    # (correspond à InvoiceLineType.MATERIAL_DELIVERY = "material_delivery")
    op.execute(
        """
        DO $$ BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_enum e
                JOIN pg_type t ON e.enumtypid = t.oid
                WHERE t.typname = 'invoice_line_type' AND e.enumlabel = 'material_delivery'
            ) THEN
                ALTER TYPE invoice_line_type ADD VALUE 'material_delivery';
            END IF;
        END $$;
        """
    )


def downgrade():
    # PostgreSQL ne permet pas de supprimer une valeur d'enum facilement
    # On laisse la valeur pour éviter des erreurs sur les lignes existantes
    pass
