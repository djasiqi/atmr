"""contact notification retry fields

Revision ID: 3d4ba7a5e8ab
Revises: f997965465f6
Create Date: 2026-09-08 13:39:43.425991

Autogenerate Alembic a détecté un écart de schéma plus large.
Cette révision ne conserve que les colonnes contact_requests nécessaires
au suivi / retry des notifications internes.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


revision = "3d4ba7a5e8ab"
down_revision = "f997965465f6"
branch_labels = None
depends_on = None

_COLUMNS = (
    (
        "autoreply_delivery_status",
        sa.Column(
            "autoreply_delivery_status",
            sa.String(length=32),
            server_default="pending",
            nullable=False,
        ),
    ),
    (
        "notification_retry_count",
        sa.Column(
            "notification_retry_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    ),
    (
        "notification_last_error",
        sa.Column("notification_last_error", sa.String(length=512), nullable=True),
    ),
    (
        "notification_last_attempt_at",
        sa.Column(
            "notification_last_attempt_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    ),
)


def upgrade():
    bind = op.get_bind()
    inspector = inspect(bind)
    if "contact_requests" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("contact_requests")}
    with op.batch_alter_table("contact_requests", schema=None) as batch_op:
        for name, column in _COLUMNS:
            if name not in existing:
                batch_op.add_column(column)


def downgrade():
    bind = op.get_bind()
    inspector = inspect(bind)
    if "contact_requests" not in inspector.get_table_names():
        return
    existing = {col["name"] for col in inspector.get_columns("contact_requests")}
    with op.batch_alter_table("contact_requests", schema=None) as batch_op:
        for name, _column in reversed(_COLUMNS):
            if name in existing:
                batch_op.drop_column(name)
