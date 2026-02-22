"""Add insurance_company_name to vehicle table.

Revision ID: 20260219_veh_ins_name
Revises: 20260219_veh_tacho
Create Date: 2026-02-19
"""

import sqlalchemy as sa
from alembic import op

revision = "20260219_veh_ins_name"
down_revision = "20260219_veh_tacho"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "vehicle",
        sa.Column("insurance_company_name", sa.String(120), nullable=True),
    )


def downgrade():
    op.drop_column("vehicle", "insurance_company_name")
