"""Fusion des branches Alembic booking/messaging (3 heads).

Revision ID: 20260526_merge_heads
Revises: 20260520_fix_dispatch_legacy, 20260526_booking_change_audit, 44aa6f34c2a5
Create Date: 2026-05-26
"""

from __future__ import annotations

revision = "20260526_merge_heads"
down_revision = (
    "20260520_fix_dispatch_legacy",
    "20260526_booking_change_audit",
    "44aa6f34c2a5",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
