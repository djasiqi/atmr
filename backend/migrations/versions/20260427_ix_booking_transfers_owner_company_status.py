"""Index composite owner_company_id, status sur booking_transfers (P15).

Revision ID: 20260427_bt_owner_status
Revises: 20260423_pl_indic
Create Date: 2026-04-27
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect, text

revision = "20260427_bt_owner_status"
down_revision = "20260423_pl_indic"
branch_labels = None
depends_on = None

INDEX_NAME = "ix_booking_transfers_owner_company_status"


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    inspector = inspect(bind)
    if "booking_transfers" not in inspector.get_table_names(schema="public"):
        return
    existing = {idx["name"] for idx in inspector.get_indexes("booking_transfers", schema="public")}
    if INDEX_NAME in existing:
        return
    op.execute(
        text(
            f"CREATE INDEX IF NOT EXISTS {INDEX_NAME} "
            "ON booking_transfers (owner_company_id, status)"
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute(text(f"DROP INDEX IF EXISTS {INDEX_NAME}"))
