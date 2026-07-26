"""Migration — table driver_device_health_events."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260601_device_health"
down_revision = "20260527_dt_company"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "driver_device_health_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "driver_id",
            sa.Integer(),
            sa.ForeignKey("driver.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("manufacturer", sa.String(length=64), nullable=True),
        sa.Column("model", sa.String(length=128), nullable=True),
        sa.Column("platform", sa.String(length=16), nullable=True),
        sa.Column("battery_optimized", sa.Boolean(), nullable=True),
        sa.Column("location_permission", sa.String(length=32), nullable=True),
        sa.Column("notifications_enabled", sa.Boolean(), nullable=True),
        sa.Column("tracking_active", sa.Boolean(), nullable=True),
        sa.Column("app_state", sa.String(length=32), nullable=True),
        sa.Column("last_fix_age_seconds", sa.Integer(), nullable=True),
        sa.Column("constraint_reason", sa.String(length=64), nullable=True),
        sa.Column("fgs_running", sa.Boolean(), nullable=True),
        sa.Column("trigger_reason", sa.String(length=128), nullable=True),
    )
    op.create_index(
        "ix_driver_device_health_events_driver_id",
        "driver_device_health_events",
        ["driver_id"],
    )
    op.create_index(
        "ix_driver_device_health_events_recorded_at",
        "driver_device_health_events",
        ["recorded_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_driver_device_health_events_recorded_at",
        table_name="driver_device_health_events",
    )
    op.drop_index(
        "ix_driver_device_health_events_driver_id",
        table_name="driver_device_health_events",
    )
    op.drop_table("driver_device_health_events")
