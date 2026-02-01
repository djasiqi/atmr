"""add CHECK constraint: material_delivery requires delivery_description

Revision ID: 20260130_material_delivery_check
Revises: 20260130_invoice_line_type
Create Date: 2026-01-30

Évite les livraisons sans description (API externe, import, bug).
"""

from alembic import op


revision = "20260130_material_delivery_check"
down_revision = "20260130_invoice_line_type"
branch_labels = None
depends_on = None


def upgrade():
    op.create_check_constraint(
        "ck_booking_material_delivery_description",
        "booking",
        "mission_type != 'material_delivery' OR delivery_description IS NOT NULL",
    )


def downgrade():
    op.drop_constraint(
        "ck_booking_material_delivery_description",
        "booking",
        type_="check",
    )
