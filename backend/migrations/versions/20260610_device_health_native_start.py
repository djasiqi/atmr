"""Migration — native FGS start diagnostics on driver_device_health_events."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260610_native_start_diag"
down_revision = "20260609_user_job_title"
branch_labels = None
depends_on = None

TABLE = "driver_device_health_events"

NATIVE_COLUMNS = (
    ("native_start_phase", sa.String(length=64)),
    ("native_start_error", sa.String(length=512)),
    ("native_task_defined", sa.Boolean()),
    ("native_started_before", sa.Boolean()),
    ("native_started_after", sa.Boolean()),
)


def _existing_columns() -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(TABLE)}


def upgrade() -> None:
    existing = _existing_columns()
    for name, column_type in NATIVE_COLUMNS:
        if name not in existing:
            op.add_column(TABLE, sa.Column(name, column_type, nullable=True))


def downgrade() -> None:
    existing = _existing_columns()
    for name, _ in reversed(NATIVE_COLUMNS):
        if name in existing:
            op.drop_column(TABLE, name)
