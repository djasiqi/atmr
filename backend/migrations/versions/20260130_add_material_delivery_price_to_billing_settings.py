"""add material_delivery_price_fixed to company_billing_settings

Revision ID: 20260130_material_delivery_price
Revises: 20260130_mission_delivery
Create Date: 2026-01-30

"""

from alembic import op
import sqlalchemy as sa


revision = "20260130_material_delivery_price"
down_revision = "20260130_mission_delivery"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("company_billing_settings", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "material_delivery_price_fixed",
                sa.Numeric(10, 2),
                nullable=True,
            )
        )


def downgrade():
    with op.batch_alter_table("company_billing_settings", schema=None) as batch_op:
        batch_op.drop_column("material_delivery_price_fixed")
