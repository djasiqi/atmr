"""Add dropoff admin freeze fields to booking.

Revision ID: 20260226_dropoff_admin
Revises: 20260226_pickup_admin
Create Date: 2026-02-26 12:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260226_dropoff_admin"
down_revision = "20260226_pickup_admin"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "booking", sa.Column("dropoff_admin_token", sa.String(length=64), nullable=True)
    )
    op.add_column(
        "booking", sa.Column("dropoff_canton_code", sa.String(length=8), nullable=True)
    )
    op.add_column(
        "booking",
        sa.Column("dropoff_admin_source", sa.String(length=24), nullable=True),
    )
    op.add_column(
        "booking",
        sa.Column("dropoff_admin_confidence", sa.String(length=24), nullable=True),
    )
    op.add_column(
        "booking",
        sa.Column("dropoff_admin_label", sa.String(length=160), nullable=True),
    )
    op.add_column(
        "booking",
        sa.Column(
            "dropoff_admin_resolved_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    op.create_index(
        op.f("ix_booking_dropoff_admin_token"),
        "booking",
        ["dropoff_admin_token"],
        unique=False,
    )
    op.create_index(
        op.f("ix_booking_dropoff_canton_code"),
        "booking",
        ["dropoff_canton_code"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_booking_dropoff_canton_code"), table_name="booking")
    op.drop_index(op.f("ix_booking_dropoff_admin_token"), table_name="booking")
    op.drop_column("booking", "dropoff_admin_resolved_at")
    op.drop_column("booking", "dropoff_admin_label")
    op.drop_column("booking", "dropoff_admin_confidence")
    op.drop_column("booking", "dropoff_admin_source")
    op.drop_column("booking", "dropoff_canton_code")
    op.drop_column("booking", "dropoff_admin_token")
