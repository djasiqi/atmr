"""Colonnes lifecycle push sur device_tokens.

Revision ID: 20260328_dt_push_lifecycle (max 32 chars pour alembic_version)
Revises: 20260326_ix_booking_created_at
Create Date: 2026-03-28

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260328_dt_push_lifecycle"
down_revision = "20260326_ix_booking_created_at"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "device_tokens",
        sa.Column(
            "last_push_success_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "device_tokens",
        sa.Column(
            "last_push_failure_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "device_tokens",
        sa.Column(
            "consecutive_push_failures",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "device_tokens",
        sa.Column("last_push_error_code", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("device_tokens", "last_push_error_code")
    op.drop_column("device_tokens", "consecutive_push_failures")
    op.drop_column("device_tokens", "last_push_failure_at")
    op.drop_column("device_tokens", "last_push_success_at")
