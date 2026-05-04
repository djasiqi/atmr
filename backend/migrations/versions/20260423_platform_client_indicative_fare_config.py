"""Indicatif portail client: table singleton de calibration (hors compute_price).

Revision ID: 20260423_pl_indic
Revises: 20260422_merge_guest_ts
Create Date: 2026-04-23
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy import text

revision = "20260423_pl_indic"
down_revision = "20260422_merge_guest_ts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "platform_client_indicative_fare_config",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column("min_fare_chf", sa.Numeric(12, 4), nullable=False),
        sa.Column("base_chf", sa.Numeric(12, 4), nullable=False),
        sa.Column("per_minute_chf", sa.Numeric(12, 4), nullable=False),
        sa.Column("ref_km", sa.Numeric(12, 4), nullable=False),
        sa.Column("ref_min", sa.Numeric(12, 4), nullable=False),
        sa.Column(
            "config_version", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
        sa.Column("calibration_note", sa.Text(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_by_user_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_plat_indic_fare_updated_by_user",
        "platform_client_indicative_fare_config",
        "user",
        ["updated_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_platform_client_indicative_fare_config_updated_by",
        "platform_client_indicative_fare_config",
        ["updated_by_user_id"],
    )
    # Seed singleton id=1 (aligné ClientDashboard: 45, 18, 0,35, 13,5, 20)
    op.execute(
        text(
            """
        INSERT INTO platform_client_indicative_fare_config (
            id, is_enabled, min_fare_chf, base_chf, per_minute_chf, ref_km, ref_min,
            config_version, calibration_note, updated_by_user_id
        ) VALUES (
            1, true,
            45, 18, 0.35, 13.5, 20,
            1, NULL, NULL
        )
    """
        )
    )


def downgrade() -> None:
    op.drop_index(
        "ix_platform_client_indicative_fare_config_updated_by",
        table_name="platform_client_indicative_fare_config",
    )
    op.drop_table("platform_client_indicative_fare_config")
