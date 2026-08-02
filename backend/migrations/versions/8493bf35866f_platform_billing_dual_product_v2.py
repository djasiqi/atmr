"""platform_billing_dual_product_v2

Facturation plateforme dual-produit : contrats versionnés, grille, créancier,
billing_origin, relevé enrichi, preuves, facture émise, paiements.

Revision ID: 8493bf35866f
Revises: 801a7e0a2923
Create Date: 2026-08-02
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "8493bf35866f"
down_revision = "801a7e0a2923"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "platform_subscription_pricing_grid",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("grid_key", sa.String(length=64), nullable=False, server_default="default"),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default="CHF"),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("valid_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
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
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_plat_sub_grid_key_active",
        "platform_subscription_pricing_grid",
        ["grid_key", "is_active"],
    )

    op.create_table(
        "platform_subscription_pricing_tier",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("grid_id", sa.Integer(), nullable=False),
        sa.Column("volume_min", sa.Integer(), nullable=False),
        sa.Column("volume_max", sa.Integer(), nullable=True),
        sa.Column("price_monthly", sa.Numeric(12, 2), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(
            ["grid_id"],
            ["platform_subscription_pricing_grid.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_plat_sub_tier_grid",
        "platform_subscription_pricing_tier",
        ["grid_id", "volume_min"],
    )

    op.create_table(
        "platform_billing_creditor",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("legal_name", sa.String(length=200), nullable=False),
        sa.Column("street_name", sa.String(length=70), nullable=False),
        sa.Column("building_number", sa.String(length=16), nullable=True),
        sa.Column("postal_code", sa.String(length=16), nullable=False),
        sa.Column("city", sa.String(length=35), nullable=False),
        sa.Column("country_code", sa.String(length=2), nullable=False, server_default="CH"),
        sa.Column("uid_ide", sa.String(length=20), nullable=True),
        sa.Column("vat_number", sa.String(length=32), nullable=True),
        sa.Column(
            "default_tax_rate",
            sa.Numeric(8, 4),
            nullable=False,
            server_default="8.1000",
        ),
        sa.Column("iban", sa.String(length=34), nullable=True),
        sa.Column("qr_iban", sa.String(length=34), nullable=True),
        sa.Column(
            "payment_reference_mode",
            sa.String(length=16),
            nullable=False,
            server_default="QRR",
        ),
        sa.Column("creditor_reference_base", sa.String(length=32), nullable=True),
        sa.Column(
            "payment_terms_days_default",
            sa.Integer(),
            nullable=False,
            server_default="30",
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="true"),
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
        sa.PrimaryKeyConstraint("id"),
    )

    # Colonnes contrat
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "own_portfolio_billing_enabled",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "lirie_commission_enabled",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "support_enabled", sa.Boolean(), nullable=False, server_default="false"
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "subscription_pricing_mode",
            sa.String(length=16),
            nullable=False,
            server_default="volume",
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column("custom_subscription_amount", sa.Numeric(12, 2), nullable=True),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "use_global_pricing_grid",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column("pricing_grid_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "commission_cancellation_policy",
            sa.String(length=32),
            nullable=False,
            server_default="exclude",
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column("payment_terms_days", sa.Integer(), nullable=True),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "amounts_are_tax_inclusive",
            sa.Boolean(),
            nullable=False,
            server_default="false",
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column("tax_rate_override", sa.Numeric(8, 4), nullable=True),
    )
    op.create_foreign_key(
        "fk_cpb_config_pricing_grid",
        "company_platform_billing_config",
        "platform_subscription_pricing_grid",
        ["pricing_grid_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.execute(
        """
        UPDATE company_platform_billing_config
        SET own_portfolio_billing_enabled = true,
            lirie_commission_enabled = true,
            support_enabled = true,
            subscription_pricing_mode = 'volume',
            use_global_pricing_grid = true
        WHERE is_billing_enabled = true
        """
    )

    # Seed grille default depuis paliers semi_auto
    op.execute(
        """
        INSERT INTO platform_subscription_pricing_grid
            (grid_key, label, currency, is_active)
        VALUES ('default', 'Grille volume LIRIE (défaut)', 'CHF', true)
        """
    )
    op.execute(
        """
        INSERT INTO platform_subscription_pricing_tier
            (grid_id, volume_min, volume_max, price_monthly, label)
        SELECT g.id, v.volume_min, v.volume_max, v.price_monthly, v.label
        FROM platform_subscription_pricing_grid g
        CROSS JOIN (VALUES
            (0, 200, 79.00::numeric, 'Palier 0–200'),
            (201, 500, 149.00::numeric, 'Palier 201–500'),
            (501, NULL::integer, 249.00::numeric, 'Palier 501+')
        ) AS v(volume_min, volume_max, price_monthly, label)
        WHERE g.grid_key = 'default'
        """
    )

    op.add_column(
        "platform_subscription_pricing",
        sa.Column("grid_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_plat_sub_pricing_grid",
        "platform_subscription_pricing",
        "platform_subscription_pricing_grid",
        ["grid_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Relevé enrichi
    op.add_column(
        "platform_invoice",
        sa.Column("tax_rate", sa.Numeric(8, 4), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("tax_amount", sa.Numeric(12, 2), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column(
            "statement_status",
            sa.String(length=32),
            nullable=False,
            server_default="DRAFT",
        ),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("calculation_version", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("contract_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("pricing_grid_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("own_portfolio_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("subscription_amount", sa.Numeric(12, 2), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("lirie_transport_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("commission_base", sa.Numeric(12, 2), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("commission_rate_snapshot", sa.Numeric(8, 6), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("commission_amount", sa.Numeric(12, 2), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("support_amount", sa.Numeric(12, 2), nullable=True),
    )
    op.add_column(
        "platform_invoice",
        sa.Column("snapshot_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_foreign_key(
        "fk_platform_invoice_contract",
        "platform_invoice",
        "company_platform_billing_config",
        ["contract_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_platform_invoice_pricing_grid",
        "platform_invoice",
        "platform_subscription_pricing_grid",
        ["pricing_grid_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.create_table(
        "platform_billing_statement_item",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("statement_id", sa.Integer(), nullable=False),
        sa.Column("item_type", sa.String(length=32), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=True),
        sa.Column("support_entry_id", sa.Integer(), nullable=True),
        sa.Column("service_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("description", sa.String(length=512), nullable=True),
        sa.Column("quantity", sa.Numeric(12, 4), nullable=True),
        sa.Column("unit_amount", sa.Numeric(12, 4), nullable=True),
        sa.Column("base_amount", sa.Numeric(12, 2), nullable=True),
        sa.Column("rate", sa.Numeric(8, 6), nullable=True),
        sa.Column("net_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("tax_rate", sa.Numeric(8, 4), nullable=True),
        sa.Column("tax_amount", sa.Numeric(12, 2), nullable=True),
        sa.Column("gross_amount", sa.Numeric(12, 2), nullable=True),
        sa.Column("eligibility_status", sa.String(length=32), nullable=True),
        sa.Column("eligibility_reason", sa.String(length=255), nullable=True),
        sa.Column(
            "source_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["booking_id"], ["booking.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["statement_id"], ["platform_invoice.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["support_entry_id"],
            ["platform_support_entry.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_plat_stmt_item_statement",
        "platform_billing_statement_item",
        ["statement_id"],
    )
    op.create_index(
        "ix_plat_stmt_item_booking",
        "platform_billing_statement_item",
        ["booking_id"],
    )

    op.create_table(
        "platform_issued_invoice",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("statement_id", sa.Integer(), nullable=True),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("invoice_number", sa.String(length=64), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="DRAFT"
        ),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default="CHF"),
        sa.Column("subtotal_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("tax_rate", sa.Numeric(8, 4), nullable=False),
        sa.Column("tax_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("total_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("qr_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("qr_reference", sa.String(length=64), nullable=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("credited_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("credit_of_invoice_id", sa.Integer(), nullable=True),
        sa.Column("pdf_storage_key", sa.String(length=512), nullable=True),
        sa.Column("pdf_checksum", sa.String(length=128), nullable=True),
        sa.Column(
            "debtor_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "creditor_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "amount_paid",
            sa.Numeric(12, 2),
            nullable=False,
            server_default="0.00",
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
        sa.ForeignKeyConstraint(
            ["company_id"], ["company.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["credit_of_invoice_id"],
            ["platform_issued_invoice.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["statement_id"], ["platform_invoice.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("invoice_number", name="uq_platform_issued_invoice_number"),
        sa.UniqueConstraint("qr_reference", name="uq_platform_issued_invoice_qr_ref"),
    )
    op.create_index(
        "uq_platform_issued_invoice_statement",
        "platform_issued_invoice",
        ["statement_id"],
        unique=True,
    )

    op.create_table(
        "platform_invoice_payment",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("issued_invoice_id", sa.Integer(), nullable=False),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("paid_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("method", sa.String(length=32), nullable=True),
        sa.Column("reference", sa.String(length=128), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["issued_invoice_id"],
            ["platform_issued_invoice.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_plat_inv_payment_invoice",
        "platform_invoice_payment",
        ["issued_invoice_id"],
    )

    op.add_column(
        "booking",
        sa.Column("billing_origin", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "booking",
        sa.Column("billing_origin_source", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "booking",
        sa.Column("billing_origin_reason", sa.String(length=512), nullable=True),
    )
    op.create_index("ix_booking_billing_origin", "booking", ["billing_origin"])

    op.create_table(
        "booking_billing_origin_audit",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("old_value", sa.String(length=32), nullable=True),
        sa.Column("new_value", sa.String(length=32), nullable=False),
        sa.Column("reason", sa.String(length=512), nullable=False),
        sa.Column("author_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["author_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["booking_id"], ["booking.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_billing_origin_audit_booking",
        "booking_billing_origin_audit",
        ["booking_id"],
    )

    # Backfill déterministe billing_origin
    op.execute(
        """
        UPDATE booking
        SET billing_origin = 'LIRIE_MARKETPLACE',
            billing_origin_source = 'BACKFILL_DETERMINISTIC',
            billing_origin_reason = 'created_via=institution_portal'
        WHERE created_via = 'institution_portal'
          AND billing_origin IS NULL
        """
    )
    op.execute(
        """
        UPDATE booking
        SET billing_origin = 'OWN_PORTFOLIO',
            billing_origin_source = 'BACKFILL_DETERMINISTIC',
            billing_origin_reason = 'created_via portfolio channel'
        WHERE created_via IN ('dispatcher', 'client_app', 'public_guest', 'api_partner', 'legacy')
          AND billing_origin IS NULL
        """
    )


def downgrade() -> None:
    op.drop_table("booking_billing_origin_audit")
    op.drop_index("ix_booking_billing_origin", table_name="booking")
    op.drop_column("booking", "billing_origin_reason")
    op.drop_column("booking", "billing_origin_source")
    op.drop_column("booking", "billing_origin")

    op.drop_table("platform_invoice_payment")
    op.drop_index(
        "uq_platform_issued_invoice_statement", table_name="platform_issued_invoice"
    )
    op.drop_table("platform_issued_invoice")
    op.drop_table("platform_billing_statement_item")

    op.drop_constraint("fk_platform_invoice_pricing_grid", "platform_invoice", type_="foreignkey")
    op.drop_constraint("fk_platform_invoice_contract", "platform_invoice", type_="foreignkey")
    for col in (
        "snapshot_json",
        "support_amount",
        "commission_amount",
        "commission_rate_snapshot",
        "commission_base",
        "lirie_transport_count",
        "subscription_amount",
        "own_portfolio_count",
        "pricing_grid_id",
        "contract_id",
        "calculation_version",
        "statement_status",
        "tax_amount",
        "tax_rate",
    ):
        op.drop_column("platform_invoice", col)

    op.drop_constraint("fk_plat_sub_pricing_grid", "platform_subscription_pricing", type_="foreignkey")
    op.drop_column("platform_subscription_pricing", "grid_id")

    op.drop_constraint("fk_cpb_config_pricing_grid", "company_platform_billing_config", type_="foreignkey")
    for col in (
        "tax_rate_override",
        "amounts_are_tax_inclusive",
        "payment_terms_days",
        "commission_cancellation_policy",
        "pricing_grid_id",
        "use_global_pricing_grid",
        "custom_subscription_amount",
        "subscription_pricing_mode",
        "support_enabled",
        "lirie_commission_enabled",
        "own_portfolio_billing_enabled",
    ):
        op.drop_column("company_platform_billing_config", col)

    op.drop_table("platform_billing_creditor")
    op.drop_table("platform_subscription_pricing_tier")
    op.drop_table("platform_subscription_pricing_grid")
