"""Élargit adresses réservation (pickup/dropoff) à 500 car. — aligné sur BookingCreateSchema.

Revision ID: 20260410_addr500
Revises: 20260410_merge_wl_await
Create Date: 2026-04-10
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260410_addr500"
down_revision = "20260410_merge_wl_await"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "booking",
        "pickup_location",
        existing_type=sa.String(length=200),
        type_=sa.String(length=500),
        existing_nullable=False,
    )
    op.alter_column(
        "booking",
        "dropoff_location",
        existing_type=sa.String(length=200),
        type_=sa.String(length=500),
        existing_nullable=False,
    )
    op.alter_column(
        "booking",
        "customer_name",
        existing_type=sa.String(length=100),
        type_=sa.String(length=200),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "booking",
        "customer_name",
        existing_type=sa.String(length=200),
        type_=sa.String(length=100),
        existing_nullable=False,
    )
    op.alter_column(
        "booking",
        "dropoff_location",
        existing_type=sa.String(length=500),
        type_=sa.String(length=200),
        existing_nullable=False,
    )
    op.alter_column(
        "booking",
        "pickup_location",
        existing_type=sa.String(length=500),
        type_=sa.String(length=200),
        existing_nullable=False,
    )
