"""add platform zone sets

Revision ID: 20260226_platform_zone_sets
Revises: 20260226_dropoff_admin
Create Date: 2026-02-26 18:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260226_platform_zone_sets"
down_revision = "20260226_dropoff_admin"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "platform_zone_set",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=120), nullable=False),
        sa.Column("scope", sa.String(length=16), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )
    op.create_index("ix_platform_zone_set_key", "platform_zone_set", ["key"], unique=False)
    op.create_index(
        "ix_platform_zone_set_active_scope",
        "platform_zone_set",
        ["is_active", "scope"],
        unique=False,
    )

    op.create_table(
        "platform_zone",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("zone_set_id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=120), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["zone_set_id"], ["platform_zone_set.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("zone_set_id", "code", name="uq_platform_zone_set_code"),
    )
    op.create_index("ix_platform_zone_zone_set", "platform_zone", ["zone_set_id"], unique=False)

    op.create_table(
        "platform_zone_membership",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("zone_set_id", sa.Integer(), nullable=False),
        sa.Column("zone_id", sa.Integer(), nullable=False),
        sa.Column("commune_token", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["zone_id"], ["platform_zone.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["zone_set_id"], ["platform_zone_set.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "zone_set_id",
            "commune_token",
            name="uq_platform_zone_membership_zone_set_commune",
        ),
    )
    op.create_index(
        "ix_platform_zone_membership_zone_set",
        "platform_zone_membership",
        ["zone_set_id"],
        unique=False,
    )
    op.create_index(
        "ix_platform_zone_membership_commune",
        "platform_zone_membership",
        ["commune_token"],
        unique=False,
    )


def downgrade():
    op.drop_index("ix_platform_zone_membership_commune", table_name="platform_zone_membership")
    op.drop_index("ix_platform_zone_membership_zone_set", table_name="platform_zone_membership")
    op.drop_table("platform_zone_membership")

    op.drop_index("ix_platform_zone_zone_set", table_name="platform_zone")
    op.drop_table("platform_zone")

    op.drop_index("ix_platform_zone_set_active_scope", table_name="platform_zone_set")
    op.drop_index("ix_platform_zone_set_key", table_name="platform_zone_set")
    op.drop_table("platform_zone_set")

