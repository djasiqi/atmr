"""platform_issued_invoice_registry_ledger

Revision ID: e1ae4a70c23a
Revises: d0e04085600f
Create Date: 2026-08-04 13:17:59.876421

"""

from alembic import op
import sqlalchemy as sa


revision = "e1ae4a70c23a"
down_revision = "d0e04085600f"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "platform_invoice_number_sequence",
        sa.Column("billing_year", sa.Integer(), nullable=False),
        sa.Column("billing_month", sa.Integer(), nullable=False),
        sa.Column("next_value", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("billing_year", "billing_month"),
    )
    op.create_table(
        "platform_invoice_due_date_change",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("issued_invoice_id", sa.Integer(), nullable=False),
        sa.Column("old_due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("new_due_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sa.String(length=512), nullable=False),
        sa.Column("change_type", sa.String(length=32), nullable=False),
        sa.Column("admin_user_id", sa.Integer(), nullable=True),
        sa.Column("old_pdf_checksum", sa.String(length=128), nullable=True),
        sa.Column("new_pdf_checksum", sa.String(length=128), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["admin_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["issued_invoice_id"],
            ["platform_issued_invoice.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_plat_due_change_invoice",
        "platform_invoice_due_date_change",
        ["issued_invoice_id"],
        unique=False,
    )

    op.add_column(
        "platform_invoice_payment",
        sa.Column(
            "entry_type",
            sa.String(length=16),
            server_default="PAYMENT",
            nullable=False,
        ),
    )
    op.add_column(
        "platform_invoice_payment",
        sa.Column("idempotency_key", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "platform_invoice_payment",
        sa.Column("reverses_payment_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_invoice_payment",
        sa.Column("reversal_reason", sa.String(length=512), nullable=True),
    )
    op.create_index(
        "uq_plat_inv_payment_idempotency",
        "platform_invoice_payment",
        ["issued_invoice_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )
    op.create_index(
        "uq_plat_inv_payment_reverses",
        "platform_invoice_payment",
        ["reverses_payment_id"],
        unique=True,
        postgresql_where=sa.text("reverses_payment_id IS NOT NULL"),
    )
    op.create_foreign_key(
        "fk_plat_inv_payment_reverses",
        "platform_invoice_payment",
        "platform_invoice_payment",
        ["reverses_payment_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "ck_plat_inv_payment_entry_type",
        "platform_invoice_payment",
        "(entry_type = 'PAYMENT' AND amount > 0 AND reverses_payment_id IS NULL)"
        " OR "
        "(entry_type = 'REVERSAL' AND amount < 0 AND reverses_payment_id IS NOT NULL)",
    )

    op.add_column(
        "platform_issued_invoice",
        sa.Column(
            "document_type",
            sa.String(length=16),
            server_default="INVOICE",
            nullable=False,
        ),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column("replaces_issued_invoice_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column("billing_year", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column("billing_month", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column("period_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column("credit_reason", sa.String(length=512), nullable=True),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column("credit_created_by_user_id", sa.Integer(), nullable=True),
    )
    op.create_index(
        "ix_platform_issued_billing_period",
        "platform_issued_invoice",
        ["billing_year", "billing_month"],
        unique=False,
    )
    op.create_index(
        "uq_platform_issued_credit_of",
        "platform_issued_invoice",
        ["credit_of_invoice_id"],
        unique=True,
        postgresql_where=sa.text("credit_of_invoice_id IS NOT NULL"),
    )
    op.create_foreign_key(
        "fk_platform_issued_credit_author",
        "platform_issued_invoice",
        "user",
        ["credit_created_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_platform_issued_replaces",
        "platform_issued_invoice",
        "platform_issued_invoice",
        ["replaces_issued_invoice_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_platform_issued_period",
        "platform_issued_invoice",
        "platform_billing_period",
        ["period_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Backfill snapshots période depuis le relevé
    op.execute(
        """
        UPDATE platform_issued_invoice AS i
        SET
            billing_year = p.billing_year,
            billing_month = p.billing_month,
            period_id = s.period_id
        FROM platform_invoice AS s
        JOIN platform_billing_period AS p ON p.id = s.period_id
        WHERE i.statement_id = s.id
          AND i.billing_year IS NULL
        """
    )

    # Seed séquence depuis les numéros existants LIRIE-YYYY-MM-NNNN
    op.execute(
        """
        INSERT INTO platform_invoice_number_sequence
            (billing_year, billing_month, next_value)
        SELECT
            CAST(split_part(invoice_number, '-', 2) AS INTEGER),
            CAST(split_part(invoice_number, '-', 3) AS INTEGER),
            MAX(
                CASE
                    WHEN split_part(invoice_number, '-', 4) ~ '^[0-9]+$'
                    THEN CAST(split_part(invoice_number, '-', 4) AS INTEGER)
                    ELSE 0
                END
            )
        FROM platform_issued_invoice
        WHERE invoice_number LIKE 'LIRIE-%'
          AND invoice_number !~ '-AV-'
          AND invoice_number !~ '-CN$'
        GROUP BY 1, 2
        ON CONFLICT (billing_year, billing_month) DO NOTHING
        """
    )

    # Dunning : statut cancelled + unique partiel
    op.drop_constraint(
        "ck_platform_dunning_event_status",
        "platform_dunning_event",
        type_="check",
    )
    op.create_check_constraint(
        "ck_platform_dunning_event_status",
        "platform_dunning_event",
        "status IN ('pending', 'sent', 'failed', 'applied', 'cancelled')",
    )
    op.drop_index(
        "uq_platform_dunning_event_invoice_type_ver",
        table_name="platform_dunning_event",
    )
    op.create_index(
        "uq_platform_dunning_event_invoice_type_ver",
        "platform_dunning_event",
        ["invoice_id", "event_type", "policy_version"],
        unique=True,
        postgresql_where=sa.text("invoice_id IS NOT NULL AND status <> 'cancelled'"),
    )
    op.drop_index(
        "uq_platform_dunning_event_case_type",
        table_name="platform_dunning_event",
    )
    op.create_index(
        "uq_platform_dunning_event_case_type",
        "platform_dunning_event",
        ["dunning_case_id", "event_type"],
        unique=True,
        postgresql_where=sa.text("invoice_id IS NULL AND status <> 'cancelled'"),
    )


def downgrade():
    op.drop_index(
        "uq_platform_dunning_event_case_type",
        table_name="platform_dunning_event",
    )
    op.create_index(
        "uq_platform_dunning_event_case_type",
        "platform_dunning_event",
        ["dunning_case_id", "event_type"],
        unique=True,
        postgresql_where=sa.text("invoice_id IS NULL"),
    )
    op.drop_index(
        "uq_platform_dunning_event_invoice_type_ver",
        table_name="platform_dunning_event",
    )
    op.create_index(
        "uq_platform_dunning_event_invoice_type_ver",
        "platform_dunning_event",
        ["invoice_id", "event_type", "policy_version"],
        unique=True,
        postgresql_where=sa.text("invoice_id IS NOT NULL"),
    )
    op.drop_constraint(
        "ck_platform_dunning_event_status",
        "platform_dunning_event",
        type_="check",
    )
    op.create_check_constraint(
        "ck_platform_dunning_event_status",
        "platform_dunning_event",
        "status IN ('pending', 'sent', 'failed', 'applied')",
    )

    op.drop_constraint(
        "fk_platform_issued_period", "platform_issued_invoice", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_platform_issued_replaces", "platform_issued_invoice", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_platform_issued_credit_author",
        "platform_issued_invoice",
        type_="foreignkey",
    )
    op.drop_index(
        "uq_platform_issued_credit_of",
        table_name="platform_issued_invoice",
        postgresql_where=sa.text("credit_of_invoice_id IS NOT NULL"),
    )
    op.drop_index(
        "ix_platform_issued_billing_period", table_name="platform_issued_invoice"
    )
    op.drop_column("platform_issued_invoice", "credit_created_by_user_id")
    op.drop_column("platform_issued_invoice", "credit_reason")
    op.drop_column("platform_issued_invoice", "period_id")
    op.drop_column("platform_issued_invoice", "billing_month")
    op.drop_column("platform_issued_invoice", "billing_year")
    op.drop_column("platform_issued_invoice", "replaces_issued_invoice_id")
    op.drop_column("platform_issued_invoice", "document_type")

    op.drop_constraint(
        "ck_plat_inv_payment_entry_type",
        "platform_invoice_payment",
        type_="check",
    )
    op.drop_constraint(
        "fk_plat_inv_payment_reverses",
        "platform_invoice_payment",
        type_="foreignkey",
    )
    op.drop_index(
        "uq_plat_inv_payment_reverses",
        table_name="platform_invoice_payment",
        postgresql_where=sa.text("reverses_payment_id IS NOT NULL"),
    )
    op.drop_index(
        "uq_plat_inv_payment_idempotency",
        table_name="platform_invoice_payment",
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )
    op.drop_column("platform_invoice_payment", "reversal_reason")
    op.drop_column("platform_invoice_payment", "reverses_payment_id")
    op.drop_column("platform_invoice_payment", "idempotency_key")
    op.drop_column("platform_invoice_payment", "entry_type")

    op.drop_index(
        "ix_plat_due_change_invoice", table_name="platform_invoice_due_date_change"
    )
    op.drop_table("platform_invoice_due_date_change")
    op.drop_table("platform_invoice_number_sequence")
