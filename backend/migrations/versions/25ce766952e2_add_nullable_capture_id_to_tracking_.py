"""add nullable capture_id to tracking location events

Revision ID: 25ce766952e2
Revises: 9b6638784019
Create Date: 2026-08-13 17:14:21.171264

Autogenerate Alembic (flask db migrate) puis ajusté : uniquement
``capture_id`` nullable + index non unique ``(driver_id, capture_id)``.
Pas de UNIQUE au premier déploiement (observation des collisions).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "25ce766952e2"
down_revision = "9b6638784019"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "driver_location_events",
        sa.Column("capture_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_dle_driver_capture",
        "driver_location_events",
        ["driver_id", "capture_id"],
        unique=False,
    )
    op.add_column(
        "tracking_ingest_events",
        sa.Column("capture_id", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_tracking_ingest_driver_capture",
        "tracking_ingest_events",
        ["driver_id", "capture_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_tracking_ingest_driver_capture",
        table_name="tracking_ingest_events",
    )
    op.drop_column("tracking_ingest_events", "capture_id")
    op.drop_index("ix_dle_driver_capture", table_name="driver_location_events")
    op.drop_column("driver_location_events", "capture_id")
