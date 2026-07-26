"""device_health diagnostic lot1 (app/os version + ios bg signals)

Revision ID: 19bfdea7a833
Revises: 79015fbd8686
Create Date: 2026-06-24 17:42:04.897400

Lot 1 diagnostic : ajoute des colonnes d'observabilité device-health
(versions app/OS + signaux background iOS) sur `driver_device_health_events`.
Toutes nullables, aucun changement de comportement.

Migration nettoyée manuellement pour ne contenir QUE ces colonnes :
l'autogenerate captait un drift de schéma non lié (dev DB divergente).
"""

from alembic import op
import sqlalchemy as sa


revision = "19bfdea7a833"
down_revision = "79015fbd8686"
branch_labels = None
depends_on = None


_TABLE = "driver_device_health_events"

_NEW_COLUMNS = (
    ("app_version", sa.String(length=32)),
    ("os_version", sa.String(length=32)),
    ("native_last_fix_age_seconds", sa.Integer()),
    ("native_task_running", sa.Boolean()),
    ("ios_accuracy_authorization", sa.String(length=16)),
    ("ios_low_power_mode", sa.Boolean()),
    ("ios_background_refresh_status", sa.String(length=16)),
)


def _existing_columns() -> set[str]:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    return {col["name"] for col in inspector.get_columns(_TABLE)}


def upgrade():
    existing = _existing_columns()
    for name, col_type in _NEW_COLUMNS:
        if name not in existing:
            op.add_column(_TABLE, sa.Column(name, col_type, nullable=True))


def downgrade():
    existing = _existing_columns()
    for name, _ in reversed(_NEW_COLUMNS):
        if name in existing:
            op.drop_column(_TABLE, name)
