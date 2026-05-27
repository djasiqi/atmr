"""DeviceToken: company_id, last_seen_at, owner CHECK, indexes.

Revision ID: 20260527_dt_company
Revises: 20260526_institution_role
Create Date: 2026-05-27
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260527_dt_company"
down_revision = "20260526_institution_role"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "device_tokens",
        sa.Column("company_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_device_tokens_company_id",
        "device_tokens",
        "company",
        ["company_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.add_column(
        "device_tokens",
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.alter_column(
        "device_tokens",
        "driver_id",
        existing_type=sa.Integer(),
        nullable=True,
    )
    op.create_check_constraint(
        "ck_device_tokens_owner_present",
        "device_tokens",
        "(driver_id IS NOT NULL) OR (company_id IS NOT NULL)",
    )
    # Dédoublonnage avant index uniques partiels (legacy sans contrainte device_id).
    # Conserve la ligne la plus récente / active par (owner, device_id).
    op.execute(
        """
        DELETE FROM device_tokens
        WHERE id IN (
            SELECT id
            FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY driver_id, device_id
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
                           PARTITION BY company_id, device_id
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
        "ix_device_tokens_company_active",
        "device_tokens",
        ["company_id", "is_active"],
        unique=False,
    )
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


def downgrade() -> None:
    op.drop_index("uq_device_tokens_company_device_id", table_name="device_tokens")
    op.drop_index("uq_device_tokens_driver_device_id", table_name="device_tokens")
    op.drop_index("ix_device_tokens_company_active", table_name="device_tokens")
    op.drop_constraint("ck_device_tokens_owner_present", "device_tokens", type_="check")
    op.alter_column(
        "device_tokens",
        "driver_id",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.drop_column("device_tokens", "last_seen_at")
    op.drop_constraint("fk_device_tokens_company_id", "device_tokens", type_="foreignkey")
    op.drop_column("device_tokens", "company_id")
