"""add_device_tokens_table

Revision ID: cd360327d324
Revises: ab24477a2d97
Create Date: 2026-01-15 03:32:05.670610

"""

from alembic import op
import sqlalchemy as sa


revision = "cd360327d324"
down_revision = "ab24477a2d97"
branch_labels = None
depends_on = None


def upgrade():
    # ✅ CORRECTIF #3: Créer la table device_tokens pour support multi-device
    op.create_table(
        "device_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("token", sa.String(length=255), nullable=False),
        sa.Column("device_id", sa.String(length=255), nullable=True),
        sa.Column("platform", sa.String(length=20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_device_tokens_driver_id", "device_tokens", ["driver_id"])
    op.create_index("ix_device_tokens_token", "device_tokens", ["token"])
    op.create_index(
        "ix_device_tokens_driver_active", "device_tokens", ["driver_id", "is_active"]
    )

    # ✅ CORRECTIF #3: Migrer les tokens existants de driver.push_token vers device_tokens
    op.execute(
        """
        INSERT INTO device_tokens (driver_id, token, platform, created_at, updated_at, is_active)
        SELECT id, push_token, 'unknown', NOW(), NOW(), true
        FROM driver
        WHERE push_token IS NOT NULL AND push_token != ''
    """
    )


def downgrade():
    op.drop_index("ix_device_tokens_driver_active", table_name="device_tokens")
    op.drop_index("ix_device_tokens_token", table_name="device_tokens")
    op.drop_index("ix_device_tokens_driver_id", table_name="device_tokens")
    op.drop_table("device_tokens")
