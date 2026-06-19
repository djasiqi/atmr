"""DeviceToken: contrainte unique (owner, device_id, provider) pour coexister expo+fcm.

Revision ID: 20260619_dt_prov_uq
Revises: 52544f40dabd
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260619_dt_prov_uq"
down_revision = "52544f40dabd"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("uq_device_tokens_driver_device_id", table_name="device_tokens")
    op.drop_index("uq_device_tokens_company_device_id", table_name="device_tokens")

    op.execute(
        """
        DELETE FROM device_tokens
        WHERE id IN (
            SELECT id
            FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY driver_id, device_id, provider
                           ORDER BY is_active DESC, updated_at DESC, id DESC
                       ) AS rn
                FROM device_tokens
                WHERE driver_id IS NOT NULL
                  AND device_id IS NOT NULL
            ) ranked
            WHERE rn > 1
        )
        """
    )
    op.execute(
        """
        DELETE FROM device_tokens
        WHERE id IN (
            SELECT id
            FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY company_id, device_id, provider
                           ORDER BY is_active DESC, updated_at DESC, id DESC
                       ) AS rn
                FROM device_tokens
                WHERE company_id IS NOT NULL
                  AND device_id IS NOT NULL
            ) ranked
            WHERE rn > 1
        )
        """
    )

    op.create_index(
        "uq_device_tokens_driver_device_provider",
        "device_tokens",
        ["driver_id", "device_id", "provider"],
        unique=True,
        postgresql_where=sa.text(
            "device_id IS NOT NULL AND driver_id IS NOT NULL"
        ),
    )
    op.create_index(
        "uq_device_tokens_company_device_provider",
        "device_tokens",
        ["company_id", "device_id", "provider"],
        unique=True,
        postgresql_where=sa.text(
            "device_id IS NOT NULL AND company_id IS NOT NULL"
        ),
    )


def downgrade() -> None:
    op.drop_index("uq_device_tokens_company_device_provider", table_name="device_tokens")
    op.drop_index("uq_device_tokens_driver_device_provider", table_name="device_tokens")
    op.create_index(
        "uq_device_tokens_driver_device_id",
        "device_tokens",
        ["driver_id", "device_id"],
        unique=True,
        postgresql_where=sa.text("device_id IS NOT NULL AND driver_id IS NOT NULL"),
    )
    op.create_index(
        "uq_device_tokens_company_device_id",
        "device_tokens",
        ["company_id", "device_id"],
        unique=True,
        postgresql_where=sa.text("device_id IS NOT NULL AND company_id IS NOT NULL"),
    )
