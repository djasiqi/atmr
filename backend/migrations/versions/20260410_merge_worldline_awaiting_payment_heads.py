"""Fusion des branches Alembic worldline checkout et awaiting client payment.

Revision ID: 20260410_merge_wl_await
Revises: 20260408_worldline_checkout, 20260410_await_pay
Create Date: 2026-04-10
"""

from __future__ import annotations

revision = "20260410_merge_wl_await"
down_revision = ("20260408_worldline_checkout", "20260410_await_pay")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
