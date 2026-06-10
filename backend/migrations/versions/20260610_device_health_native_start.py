"""Migration — native FGS start diagnostics on driver_device_health_events."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260610_native_start_diag"
down_revision = "20260609_user_job_title"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "driver_device_health_events",
        sa.Column("native_start_phase", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "driver_device_health_events",
        sa.Column("native_start_error", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "driver_device_health_events",
        sa.Column("native_task_defined", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "driver_device_health_events",
        sa.Column("native_started_before", sa.Boolean(), nullable=True),
    )
    op.add_column(
        "driver_device_health_events",
        sa.Column("native_started_after", sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("driver_device_health_events", "native_started_after")
    op.drop_column("driver_device_health_events", "native_started_before")
    op.drop_column("driver_device_health_events", "native_task_defined")
    op.drop_column("driver_device_health_events", "native_start_error")
    op.drop_column("driver_device_health_events", "native_start_phase")
