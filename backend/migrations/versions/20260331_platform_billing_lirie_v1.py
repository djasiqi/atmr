"""Tables facturation plateforme LIRIE V1 (période, relevé, lignes, config, support).

Revision ID: 20260331_plat_bill_v1
Revises: 20260329_plat_admin_perm
Create Date: 2026-03-31

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260331_plat_bill_v1"
down_revision = "20260329_plat_admin_perm"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "platform_billing_period",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("billing_year", sa.Integer(), nullable=False),
        sa.Column("billing_month", sa.Integer(), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default="draft", nullable=False
        ),
        sa.Column(
            "timezone",
            sa.String(length=64),
            server_default="Europe/Zurich",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_platform_billing_period")),
        sa.UniqueConstraint(
            "billing_year",
            "billing_month",
            name="uq_platform_billing_period_ym",
        ),
    )
    op.create_table(
        "platform_subscription_pricing",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dispatch_mode", sa.String(length=16), nullable=False),
        sa.Column("volume_min", sa.Integer(), nullable=False),
        sa.Column("volume_max", sa.Integer(), nullable=True),
        sa.Column("price_monthly", sa.Numeric(12, 2), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_platform_subscription_pricing")),
    )
    op.create_index(
        op.f("ix_platform_sub_pricing_dispatch"),
        "platform_subscription_pricing",
        ["dispatch_mode"],
        unique=False,
    )
    op.create_table(
        "company_platform_billing_config",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column(
            "is_billing_enabled", sa.Boolean(), server_default="false", nullable=False
        ),
        sa.Column("dispatch_mode_override", sa.String(length=16), nullable=True),
        sa.Column("commission_rate", sa.Numeric(8, 6), nullable=True),
        sa.Column("support_hourly_rate_default", sa.Numeric(12, 2), nullable=True),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("effective_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["company.id"],
            name=op.f("fk_cpb_config_company"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_company_platform_billing_config")),
    )
    op.create_index(
        op.f("ix_cpb_config_company_id"),
        "company_platform_billing_config",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_cpb_config_company_active"),
        "company_platform_billing_config",
        ["company_id", "is_active"],
        unique=False,
    )
    op.create_table(
        "platform_invoice",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("period_id", sa.Integer(), nullable=False),
        sa.Column(
            "currency", sa.String(length=3), server_default="CHF", nullable=False
        ),
        sa.Column("subtotal_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("total_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["company.id"],
            name=op.f("fk_platform_invoice_company"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["period_id"],
            ["platform_billing_period.id"],
            name=op.f("fk_platform_invoice_period"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_platform_invoice")),
        sa.UniqueConstraint(
            "company_id",
            "period_id",
            name="uq_platform_invoice_company_period",
        ),
    )
    op.create_index(
        op.f("ix_platform_invoice_company_id"),
        "platform_invoice",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_platform_invoice_period_id"),
        "platform_invoice",
        ["period_id"],
        unique=False,
    )
    op.create_table(
        "platform_invoice_line",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("invoice_id", sa.Integer(), nullable=False),
        sa.Column("line_type", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=True),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("quantity", sa.Numeric(12, 4), nullable=True),
        sa.Column("unit_amount", sa.Numeric(12, 4), nullable=True),
        sa.Column(
            "snapshot_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("sort_order", sa.Integer(), server_default="0", nullable=False),
        sa.ForeignKeyConstraint(
            ["invoice_id"],
            ["platform_invoice.id"],
            name=op.f("fk_platform_invoice_line_invoice"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_platform_invoice_line")),
    )
    op.create_index(
        op.f("ix_platform_invoice_line_invoice_id"),
        "platform_invoice_line",
        ["invoice_id"],
        unique=False,
    )
    op.create_table(
        "platform_support_entry",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_minutes", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("hourly_rate_snapshot", sa.Numeric(12, 2), nullable=False),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("validated_by_user_id", sa.Integer(), nullable=True),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("billing_period_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["billing_period_id"],
            ["platform_billing_period.id"],
            name=op.f("fk_platform_support_billing_period"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["company.id"],
            name=op.f("fk_platform_support_company"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["validated_by_user_id"],
            ["user.id"],
            name=op.f("fk_platform_support_validated_by"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_platform_support_entry")),
    )
    op.create_index(
        op.f("ix_platform_support_company"),
        "platform_support_entry",
        ["company_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_platform_support_company"), table_name="platform_support_entry"
    )
    op.drop_table("platform_support_entry")
    op.drop_index(
        op.f("ix_platform_invoice_line_invoice_id"), table_name="platform_invoice_line"
    )
    op.drop_table("platform_invoice_line")
    op.drop_index(op.f("ix_platform_invoice_period_id"), table_name="platform_invoice")
    op.drop_index(op.f("ix_platform_invoice_company_id"), table_name="platform_invoice")
    op.drop_table("platform_invoice")
    op.drop_index(
        op.f("ix_cpb_config_company_active"),
        table_name="company_platform_billing_config",
    )
    op.drop_index(
        op.f("ix_cpb_config_company_id"), table_name="company_platform_billing_config"
    )
    op.drop_table("company_platform_billing_config")
    op.drop_index(
        op.f("ix_platform_sub_pricing_dispatch"),
        table_name="platform_subscription_pricing",
    )
    op.drop_table("platform_subscription_pricing")
    op.drop_table("platform_billing_period")
