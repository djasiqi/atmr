"""Fusion des branches guest checkout et Timescale trip_tracking.

Revision ID: 20260422_merge_guest_ts
Revises: 20260422_guest_ck, p2_timescaledb_tracking
Create Date: 2026-04-22
"""

from __future__ import annotations

revision = "20260422_merge_guest_ts"
down_revision = ("20260422_guest_ck", "p2_timescaledb_tracking")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
