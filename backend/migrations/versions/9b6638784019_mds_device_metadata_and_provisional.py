"""mds_device_metadata_and_provisional

Revision ID: 9b6638784019
Revises: (autodetected head at generation time)
Create Date: 2026-08-10

Colonnes metadata appareil + provisional confirmation pour MobileDeviceSession.
Backfill confirmed_at pour les sessions existantes.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "9b6638784019"
down_revision = "fb24f96be76e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "mobile_device_session",
        sa.Column("device_model", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "mobile_device_session",
        sa.Column("device_manufacturer", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "mobile_device_session",
        sa.Column("device_type", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "mobile_device_session",
        sa.Column("last_app_build", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "mobile_device_session",
        sa.Column("last_os_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "mobile_device_session",
        sa.Column("metadata_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "mobile_device_session",
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "mobile_device_session",
        sa.Column("provisional_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_mobile_device_session_provisional_expires",
        "mobile_device_session",
        ["provisional_expires_at"],
        unique=False,
        postgresql_where=sa.text(
            "status = 'active' AND confirmed_at IS NULL"
        ),
    )
    # Backfill : sessions existantes considérées confirmées
    op.execute(
        """
        UPDATE mobile_device_session
        SET confirmed_at = COALESCE(last_refresh_at, last_seen_at, created_at),
            provisional_expires_at = NULL
        WHERE confirmed_at IS NULL
        """
    )


def downgrade() -> None:
    op.drop_index(
        "ix_mobile_device_session_provisional_expires",
        table_name="mobile_device_session",
    )
    op.drop_column("mobile_device_session", "provisional_expires_at")
    op.drop_column("mobile_device_session", "confirmed_at")
    op.drop_column("mobile_device_session", "metadata_updated_at")
    op.drop_column("mobile_device_session", "last_os_version")
    op.drop_column("mobile_device_session", "last_app_build")
    op.drop_column("mobile_device_session", "device_type")
    op.drop_column("mobile_device_session", "device_manufacturer")
    op.drop_column("mobile_device_session", "device_model")
