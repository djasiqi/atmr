"""add tracking_pipeline jsonb to driver_device_health_events

Revision ID: 14d1b170291f
Revises: ad0020bf5f62
Create Date: 2026-08-27 11:25:36.735195

JZ-R1 : snapshot pipeline tracking (instrumentation remote-first, nullable).
Autogenerate filtré : uniquement cette colonne (drift global exclu).
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "14d1b170291f"
down_revision = "ad0020bf5f62"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "driver_device_health_events",
        sa.Column(
            "tracking_pipeline",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade():
    op.drop_column("driver_device_health_events", "tracking_pipeline")
