"""Add vehicle_id foreign key to driver table

Revision ID: 20260218_drv_vehicle
Revises: 20260217_drv_identity
Create Date: 2026-02-18
"""

from alembic import op
import sqlalchemy as sa

revision = "20260218_drv_vehicle"
down_revision = "20260217_drv_identity"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("driver", sa.Column("vehicle_id", sa.Integer(), nullable=True))
    op.create_index("ix_driver_vehicle_id", "driver", ["vehicle_id"])
    op.create_foreign_key(
        "fk_driver_vehicle_id",
        "driver",
        "vehicle",
        ["vehicle_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    op.drop_constraint("fk_driver_vehicle_id", "driver", type_="foreignkey")
    op.drop_index("ix_driver_vehicle_id", table_name="driver")
    op.drop_column("driver", "vehicle_id")
