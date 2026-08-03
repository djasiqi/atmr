"""platform_billing_dunning_v1

Revision ID: e4f273565844
Revises: c5621b2b3dc2
Create Date: 2026-08-03 00:15:22.363478

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "e4f273565844"
down_revision = "c5621b2b3dc2"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "platform_dunning_case",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default="open", nullable=False
        ),
        sa.Column(
            "policy_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "opened_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("partial_suspended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("full_suspended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("trigger_invoice_id", sa.Integer(), nullable=True),
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
        sa.CheckConstraint(
            "status IN ('open', 'partial', 'full', 'resolved')",
            name="ck_platform_dunning_case_status",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"], ["company.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["trigger_invoice_id"],
            ["platform_issued_invoice.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_platform_dunning_case_company",
        "platform_dunning_case",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        "uq_platform_dunning_case_active",
        "platform_dunning_case",
        ["company_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('open', 'partial', 'full')"),
    )

    op.create_table(
        "platform_invoice_dunning_hold",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("issued_invoice_id", sa.Integer(), nullable=False),
        sa.Column("reason", sa.String(length=512), nullable=False),
        sa.Column("disputed_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("hold_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("released_at", sa.DateTime(timezone=True), nullable=True),
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
        "ix_platform_dunning_hold_invoice",
        "platform_invoice_dunning_hold",
        ["issued_invoice_id"],
        unique=False,
    )

    op.create_table(
        "platform_dunning_event",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("dunning_case_id", sa.Integer(), nullable=False),
        sa.Column("invoice_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default="pending", nullable=False
        ),
        sa.Column("policy_version", sa.Integer(), server_default="1", nullable=False),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider_message_id", sa.String(length=128), nullable=True),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
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
        sa.CheckConstraint(
            "status IN ('pending', 'sent', 'failed', 'applied')",
            name="ck_platform_dunning_event_status",
        ),
        sa.ForeignKeyConstraint(
            ["dunning_case_id"],
            ["platform_dunning_case.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["invoice_id"],
            ["platform_issued_invoice.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_platform_dunning_event_case",
        "platform_dunning_event",
        ["dunning_case_id"],
        unique=False,
    )
    op.create_index(
        "ix_platform_dunning_event_status",
        "platform_dunning_event",
        ["status"],
        unique=False,
    )
    op.create_index(
        "uq_platform_dunning_event_case_type",
        "platform_dunning_event",
        ["dunning_case_id", "event_type"],
        unique=True,
        postgresql_where=sa.text("invoice_id IS NULL"),
    )
    op.create_index(
        "uq_platform_dunning_event_invoice_type_ver",
        "platform_dunning_event",
        ["invoice_id", "event_type", "policy_version"],
        unique=True,
        postgresql_where=sa.text("invoice_id IS NOT NULL"),
    )

    op.add_column(
        "company",
        sa.Column(
            "platform_billing_access_state",
            sa.String(length=16),
            server_default="active",
            nullable=False,
            comment="active|partial|full — mode restreint commercial",
        ),
    )
    op.add_column(
        "company",
        sa.Column(
            "platform_billing_state_source",
            sa.String(length=32),
            nullable=True,
            comment="automatic_dunning|admin_manual",
        ),
    )
    op.add_column(
        "company",
        sa.Column("platform_billing_state_reason_code", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "company",
        sa.Column("platform_billing_state_since", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "company",
        sa.Column("platform_billing_state_config_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "company",
        sa.Column(
            "platform_billing_state_updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.add_column(
        "company",
        sa.Column("dunning_paused_until", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "company",
        sa.Column("dunning_pause_reason", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "company",
        sa.Column("dunning_paused_by_user_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_company_dunning_paused_by_user",
        "company",
        "user",
        ["dunning_paused_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "ck_company_billing_access_state",
        "company",
        "platform_billing_access_state IN ('active', 'partial', 'full')",
    )
    op.create_check_constraint(
        "ck_company_billing_state_source",
        "company",
        "platform_billing_state_source IS NULL OR "
        "platform_billing_state_source IN ('automatic_dunning', 'admin_manual')",
    )
    op.create_check_constraint(
        "ck_company_billing_state_fields",
        "company",
        "(platform_billing_access_state = 'active') OR "
        "(platform_billing_state_source IS NOT NULL AND "
        "platform_billing_state_reason_code IS NOT NULL AND "
        "platform_billing_state_since IS NOT NULL)",
    )

    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "automated_dunning_enabled",
            sa.Boolean(),
            server_default="true",
            nullable=False,
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "reminder_delay_days_after_due",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "reminder_grace_days",
            sa.Integer(),
            server_default="10",
            nullable=False,
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "full_suspend_days_after_due",
            sa.Integer(),
            server_default="30",
            nullable=False,
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "full_suspend_overdue_invoice_count",
            sa.Integer(),
            server_default="2",
            nullable=False,
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "termination_notice_days",
            sa.Integer(),
            server_default="10",
            nullable=False,
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "partial_block_marketplace_offers",
            sa.Boolean(),
            server_default="true",
            nullable=False,
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "partial_block_marketplace_acceptance",
            sa.Boolean(),
            server_default="true",
            nullable=False,
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "partial_block_billable_support",
            sa.Boolean(),
            server_default="true",
            nullable=False,
        ),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "partial_block_billable_configuration",
            sa.Boolean(),
            server_default="true",
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "ck_cpb_reminder_delay",
        "company_platform_billing_config",
        "reminder_delay_days_after_due BETWEEN 0 AND 30",
    )
    op.create_check_constraint(
        "ck_cpb_reminder_grace",
        "company_platform_billing_config",
        "reminder_grace_days BETWEEN 1 AND 30",
    )
    op.create_check_constraint(
        "ck_cpb_full_suspend_days",
        "company_platform_billing_config",
        "full_suspend_days_after_due BETWEEN 7 AND 90",
    )
    op.create_check_constraint(
        "ck_cpb_full_suspend_count",
        "company_platform_billing_config",
        "full_suspend_overdue_invoice_count BETWEEN 1 AND 12",
    )
    op.create_check_constraint(
        "ck_cpb_termination_notice",
        "company_platform_billing_config",
        "termination_notice_days BETWEEN 1 AND 30",
    )
    op.create_check_constraint(
        "ck_cpb_full_after_grace",
        "company_platform_billing_config",
        "full_suspend_days_after_due > "
        "(reminder_delay_days_after_due + reminder_grace_days)",
    )

    op.add_column(
        "platform_issued_invoice",
        sa.Column("billing_config_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column("partner_agreement_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column(
            "dunning_policy_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column(
            "dunning_automation_authorized_at_issuance",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )
    op.create_index(
        "ix_platform_issued_invoice_billing_config_id",
        "platform_issued_invoice",
        ["billing_config_id"],
        unique=False,
    )
    op.create_index(
        "ix_platform_issued_invoice_partner_agreement_id",
        "platform_issued_invoice",
        ["partner_agreement_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_pii_billing_config",
        "platform_issued_invoice",
        "company_platform_billing_config",
        ["billing_config_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_pii_partner_agreement",
        "platform_issued_invoice",
        "platform_partner_agreement",
        ["partner_agreement_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade():
    op.drop_constraint(
        "fk_pii_partner_agreement", "platform_issued_invoice", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_pii_billing_config", "platform_issued_invoice", type_="foreignkey"
    )
    op.drop_index(
        "ix_platform_issued_invoice_partner_agreement_id",
        table_name="platform_issued_invoice",
    )
    op.drop_index(
        "ix_platform_issued_invoice_billing_config_id",
        table_name="platform_issued_invoice",
    )
    op.drop_column("platform_issued_invoice", "dunning_automation_authorized_at_issuance")
    op.drop_column("platform_issued_invoice", "dunning_policy_snapshot")
    op.drop_column("platform_issued_invoice", "partner_agreement_id")
    op.drop_column("platform_issued_invoice", "billing_config_id")

    op.drop_constraint(
        "ck_cpb_full_after_grace", "company_platform_billing_config", type_="check"
    )
    op.drop_constraint(
        "ck_cpb_termination_notice", "company_platform_billing_config", type_="check"
    )
    op.drop_constraint(
        "ck_cpb_full_suspend_count", "company_platform_billing_config", type_="check"
    )
    op.drop_constraint(
        "ck_cpb_full_suspend_days", "company_platform_billing_config", type_="check"
    )
    op.drop_constraint(
        "ck_cpb_reminder_grace", "company_platform_billing_config", type_="check"
    )
    op.drop_constraint(
        "ck_cpb_reminder_delay", "company_platform_billing_config", type_="check"
    )
    for col in (
        "partial_block_billable_configuration",
        "partial_block_billable_support",
        "partial_block_marketplace_acceptance",
        "partial_block_marketplace_offers",
        "termination_notice_days",
        "full_suspend_overdue_invoice_count",
        "full_suspend_days_after_due",
        "reminder_grace_days",
        "reminder_delay_days_after_due",
        "automated_dunning_enabled",
    ):
        op.drop_column("company_platform_billing_config", col)

    op.drop_constraint(
        "ck_company_billing_state_fields", "company", type_="check"
    )
    op.drop_constraint(
        "ck_company_billing_state_source", "company", type_="check"
    )
    op.drop_constraint(
        "ck_company_billing_access_state", "company", type_="check"
    )
    op.drop_constraint(
        "fk_company_dunning_paused_by_user", "company", type_="foreignkey"
    )
    for col in (
        "dunning_paused_by_user_id",
        "dunning_pause_reason",
        "dunning_paused_until",
        "platform_billing_state_updated_at",
        "platform_billing_state_config_id",
        "platform_billing_state_since",
        "platform_billing_state_reason_code",
        "platform_billing_state_source",
        "platform_billing_access_state",
    ):
        op.drop_column("company", col)

    op.drop_index(
        "uq_platform_dunning_event_invoice_type_ver",
        table_name="platform_dunning_event",
        postgresql_where=sa.text("invoice_id IS NOT NULL"),
    )
    op.drop_index(
        "uq_platform_dunning_event_case_type",
        table_name="platform_dunning_event",
        postgresql_where=sa.text("invoice_id IS NULL"),
    )
    op.drop_index("ix_platform_dunning_event_status", table_name="platform_dunning_event")
    op.drop_index("ix_platform_dunning_event_case", table_name="platform_dunning_event")
    op.drop_table("platform_dunning_event")
    op.drop_index(
        "ix_platform_dunning_hold_invoice", table_name="platform_invoice_dunning_hold"
    )
    op.drop_table("platform_invoice_dunning_hold")
    op.drop_index(
        "uq_platform_dunning_case_active",
        table_name="platform_dunning_case",
        postgresql_where=sa.text("status IN ('open', 'partial', 'full')"),
    )
    op.drop_index(
        "ix_platform_dunning_case_company", table_name="platform_dunning_case"
    )
    op.drop_table("platform_dunning_case")
