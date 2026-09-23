"""portal receivable dunning workflow

Revision ID: 4dff22f12b4f
Revises: 919313ab962c
Create Date: 2026-09-23 23:38:03.736382

Tables générées par ``flask db revision --autogenerate``.
Écarts hors périmètre retirés. Pas de ``batch_alter_table``.
"""

import sqlalchemy as sa
from alembic import op

revision = "4dff22f12b4f"
down_revision = "919313ab962c"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "portal_receivable_dunning_policy",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("first_reminder_days", sa.Integer(), nullable=False),
        sa.Column("second_reminder_days", sa.Integer(), nullable=False),
        sa.Column("formal_notice_days", sa.Integer(), nullable=False),
        sa.Column("charge_default_interest", sa.Boolean(), nullable=False),
        sa.Column(
            "default_interest_rate", sa.Numeric(precision=6, scale=4), nullable=True
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
        sa.CheckConstraint(
            "first_reminder_days >= 0", name="ck_portal_dunning_first_nonneg"
        ),
        sa.CheckConstraint(
            "formal_notice_days >= second_reminder_days",
            name="ck_portal_dunning_formal_ge_second",
        ),
        sa.CheckConstraint(
            "second_reminder_days >= first_reminder_days",
            name="ck_portal_dunning_second_ge_first",
        ),
        sa.ForeignKeyConstraint(
            ["company_id"], ["company.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "company_id", name="uq_portal_dunning_policy_company"
        ),
    )
    op.create_table(
        "portal_receivable_dunning_event",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("receivable_id", sa.Integer(), nullable=False),
        sa.Column("creditor_company_id", sa.Integer(), nullable=False),
        sa.Column("debtor_user_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=40), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("channel", sa.String(length=32), nullable=False),
        sa.Column("recipient", sa.String(length=255), nullable=True),
        sa.Column("recipient_email", sa.String(length=255), nullable=True),
        sa.Column("template_version", sa.String(length=64), nullable=False),
        sa.Column("rendered_subject", sa.String(length=500), nullable=True),
        sa.Column("rendered_body", sa.Text(), nullable=False),
        sa.Column("rendered_body_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "balance_due_snapshot", sa.Numeric(precision=12, scale=2), nullable=False
        ),
        sa.Column("due_date_snapshot", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "external_invoice_number_snapshot", sa.String(length=80), nullable=False
        ),
        sa.Column("delivery_status", sa.String(length=32), nullable=False),
        sa.Column("provider_message_id", sa.String(length=200), nullable=True),
        sa.Column("delivery_error", sa.Text(), nullable=True),
        sa.Column("initiated_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "channel IN ('email', 'letter_draft', 'internal')",
            name="ck_portal_dunning_channel",
        ),
        sa.CheckConstraint(
            "delivery_status IN ('sent', 'failed', 'draft', 'recorded')",
            name="ck_portal_dunning_delivery",
        ),
        sa.CheckConstraint(
            "event_type IN ("
            "'REMINDER_1', 'REMINDER_2', 'FORMAL_NOTICE', 'COLLECTION_PREPARED'"
            ")",
            name="ck_portal_dunning_event_type",
        ),
        sa.ForeignKeyConstraint(
            ["creditor_company_id"], ["company.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["debtor_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["initiated_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["receivable_id"], ["portal_receivable.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portal_dunning_event_creditor",
        "portal_receivable_dunning_event",
        ["creditor_company_id"],
        unique=False,
    )
    op.create_index(
        "ix_portal_dunning_event_receivable",
        "portal_receivable_dunning_event",
        ["receivable_id"],
        unique=False,
    )
    op.create_index(
        "ix_portal_dunning_event_type",
        "portal_receivable_dunning_event",
        ["event_type"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_portal_dunning_event_type",
        table_name="portal_receivable_dunning_event",
    )
    op.drop_index(
        "ix_portal_dunning_event_receivable",
        table_name="portal_receivable_dunning_event",
    )
    op.drop_index(
        "ix_portal_dunning_event_creditor",
        table_name="portal_receivable_dunning_event",
    )
    op.drop_table("portal_receivable_dunning_event")
    op.drop_table("portal_receivable_dunning_policy")
