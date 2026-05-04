"""Add tachograph_expires_at to vehicle table

Revision ID: 20260219_veh_tacho
Revises: 20260218_drv_vehicle
Create Date: 2026-02-19
"""

from alembic import op
import sqlalchemy as sa

revision = "20260219_veh_tacho"
down_revision = "20260218_cancel_policy"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "vehicle",
        sa.Column("tachograph_expires_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade():
    op.drop_column("vehicle", "tachograph_expires_at")
