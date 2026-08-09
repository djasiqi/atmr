"""add_device_health_adoption_columns

Revision ID: 4490a30d6a68
Revises: a6a422986202
Create Date: 2026-08-09 23:53:33.461430

Colonnes d'adoption mobile (build natif + OTA) sur driver_device_health_events.
Autogenerate filtré : uniquement ces ajouts (drift global exclu).
"""

from alembic import op
import sqlalchemy as sa


revision = "4490a30d6a68"
down_revision = "a6a422986202"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "driver_device_health_events",
        sa.Column("native_build_version", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "driver_device_health_events",
        sa.Column("expo_runtime_version", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "driver_device_health_events",
        sa.Column("ota_update_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "driver_device_health_events",
        sa.Column("release_channel", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "driver_device_health_events",
        sa.Column("release_sha", sa.String(length=64), nullable=True),
    )


def downgrade():
    op.drop_column("driver_device_health_events", "release_sha")
    op.drop_column("driver_device_health_events", "release_channel")
    op.drop_column("driver_device_health_events", "ota_update_id")
    op.drop_column("driver_device_health_events", "expo_runtime_version")
    op.drop_column("driver_device_health_events", "native_build_version")
