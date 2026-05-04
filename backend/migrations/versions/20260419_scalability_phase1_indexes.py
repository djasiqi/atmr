"""Indexes critiques phase 1 (tracking + dispatch).

Revision ID: 20260419_scale_idx
Revises: 20260414_bmsg_client
Create Date: 2026-04-19
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

revision = "20260419_scale_idx"
down_revision = "20260414_bmsg_client"
branch_labels = None
depends_on = None


def _table_exists(inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names(schema="public")


def _columns_exist(inspector, table_name: str, columns: tuple[str, ...]) -> bool:
    if not _table_exists(inspector, table_name):
        return False
    available = {
        column.get("name")
        for column in inspector.get_columns(table_name, schema="public")
    }
    return all(column in available for column in columns)


def _create_indexes_postgres() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    with op.get_context().autocommit_block():
        if _columns_exist(inspector, "trip_tracking", ("driver_id", "recorded_at")):
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trip_tracking_driver_recorded "
                "ON trip_tracking(driver_id, recorded_at DESC)"
            )
        if _columns_exist(inspector, "driver", ("company_id", "is_active")):
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_driver_company_active "
                "ON driver(company_id, is_active) WHERE is_active = TRUE"
            )
        if _columns_exist(inspector, "assignment", ("driver_id", "status")):
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_assignment_driver_status "
                "ON assignment(driver_id, status) WHERE status NOT IN ('COMPLETED', 'CANCELLED')"
            )
        if _columns_exist(inspector, "booking", ("scheduled_at", "company_id")):
            op.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_booking_scheduled_at_company "
                "ON booking(scheduled_at DESC, company_id)"
            )


def _drop_indexes_postgres() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_booking_scheduled_at_company")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_assignment_driver_status")
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_driver_company_active")
        op.execute(
            "DROP INDEX CONCURRENTLY IF EXISTS idx_trip_tracking_driver_recorded"
        )


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        _create_indexes_postgres()


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        _drop_indexes_postgres()
