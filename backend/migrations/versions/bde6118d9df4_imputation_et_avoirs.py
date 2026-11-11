"""imputation et avoirs

Revision ID: bde6118d9df4
Revises: 393ac737dd28
Create Date: 2026-11-01 16:53:13.479641

Généré par Alembic sur une base estampillée à la révision de caisse,
puis limité à l'imputation et aux avoirs. Le reste recréait des tables
déjà présentes dans la chaîne et ne fait pas partie de ce lot.
Le déclencheur d'entreprise, absent de l'autogénération, protège
l'imputation d'une collection vers la facture d'une autre entreprise.
"""

from alembic import op
import sqlalchemy as sa


revision = "bde6118d9df4"
down_revision = "393ac737dd28"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "invoices",
        sa.Column(
            "refundable_chf",
            sa.Numeric(precision=10, scale=2),
            server_default=sa.text("0"),
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "chk_invoice_refundable_nonneg",
        "invoices",
        "refundable_chf >= 0",
    )
    op.create_unique_constraint(
        "uq_invoice_id_company", "invoices", ["id", "company_id"]
    )
    op.create_unique_constraint(
        "uq_cash_collection_id_company",
        "cash_collection",
        ["id", "company_id"],
    )
    op.add_column(
        "invoice_payments",
        sa.Column("cash_collection_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_invoice_payments_cash_collection",
        "invoice_payments",
        "cash_collection",
        ["cash_collection_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "uq_invoice_payments_cash_collection",
        "invoice_payments",
        ["cash_collection_id"],
        unique=True,
        postgresql_where=sa.text("cash_collection_id IS NOT NULL"),
    )
    op.execute(
        """
        CREATE FUNCTION reject_payment_other_company()
        RETURNS trigger
        LANGUAGE plpgsql
        AS $$
        BEGIN
          IF NEW.cash_collection_id IS NULL THEN
            RETURN NEW;
          END IF;
          IF NOT EXISTS (
            SELECT 1
            FROM cash_collection AS collection
            JOIN invoices AS invoice ON invoice.id = NEW.invoice_id
            WHERE collection.id = NEW.cash_collection_id
              AND collection.company_id = invoice.company_id
          ) THEN
            RAISE EXCEPTION 'imputation_autre_entreprise'
              USING ERRCODE = '23514';
          END IF;
          RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_invoice_payment_same_company
        BEFORE INSERT OR UPDATE OF cash_collection_id, invoice_id
        ON invoice_payments
        FOR EACH ROW
        EXECUTE FUNCTION reject_payment_other_company()
        """
    )
    op.create_table(
        "invoice_credit_sequence",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column(
            "last_value",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.CheckConstraint("last_value >= 0", name="ck_invoice_credit_sequence_nonneg"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "company_id",
            "year",
            name="uq_invoice_credit_sequence_company_year",
        ),
    )
    op.create_table(
        "invoice_credit_note",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("invoice_id", sa.Integer(), nullable=False),
        sa.Column("credit_number", sa.String(length=40), nullable=False),
        sa.Column("amount_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'issued'"),
            nullable=False,
        ),
        sa.Column(
            "issued_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("created_by_user_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("pdf_url", sa.String(length=500), nullable=True),
        sa.CheckConstraint(
            "credit_number ~ ('^AV-' || company_id::text || '-[0-9]{4}-[0-9]+$')",
            name="ck_invoice_credit_note_number_format",
        ),
        sa.CheckConstraint("status = 'issued'", name="ck_invoice_credit_note_status"),
        sa.CheckConstraint("amount_chf > 0", name="ck_invoice_credit_note_amount"),
        sa.CheckConstraint(
            "char_length(idempotency_key) > 0",
            name="ck_invoice_credit_note_key",
        ),
        sa.CheckConstraint(
            "char_length(reason) > 0", name="ck_invoice_credit_note_reason"
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["invoice_id", "company_id"],
            ["invoices.id", "invoices.company_id"],
            name="fk_invoice_credit_note_invoice_company",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "company_id", "credit_number", name="uq_invoice_credit_note_number"
        ),
        sa.UniqueConstraint(
            "company_id",
            "idempotency_key",
            name="uq_invoice_credit_note_idempotency",
        ),
    )
    op.create_index(
        "ix_invoice_credit_note_invoice",
        "invoice_credit_note",
        ["company_id", "invoice_id"],
        unique=False,
    )
    op.create_table(
        "invoice_refund",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("invoice_id", sa.Integer(), nullable=False),
        sa.Column("amount_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("cash_collection_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("amount_chf > 0", name="ck_invoice_refund_amount"),
        sa.CheckConstraint(
            "char_length(idempotency_key) > 0", name="ck_invoice_refund_key"
        ),
        sa.ForeignKeyConstraint(
            ["cash_collection_id", "company_id"],
            ["cash_collection.id", "cash_collection.company_id"],
            name="fk_invoice_refund_collection_company",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["invoice_id", "company_id"],
            ["invoices.id", "invoices.company_id"],
            name="fk_invoice_refund_invoice_company",
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "company_id",
            "idempotency_key",
            name="uq_invoice_refund_idempotency",
        ),
    )


def downgrade():
    op.drop_table("invoice_refund")
    op.drop_index("ix_invoice_credit_note_invoice", table_name="invoice_credit_note")
    op.drop_table("invoice_credit_note")
    op.drop_table("invoice_credit_sequence")
    op.execute("DROP TRIGGER trg_invoice_payment_same_company ON invoice_payments")
    op.execute("DROP FUNCTION reject_payment_other_company()")
    op.drop_index(
        "uq_invoice_payments_cash_collection",
        table_name="invoice_payments",
        postgresql_where=sa.text("cash_collection_id IS NOT NULL"),
    )
    op.drop_constraint(
        "fk_invoice_payments_cash_collection",
        "invoice_payments",
        type_="foreignkey",
    )
    op.drop_column("invoice_payments", "cash_collection_id")
    op.drop_constraint(
        "uq_cash_collection_id_company", "cash_collection", type_="unique"
    )
    op.drop_constraint("uq_invoice_id_company", "invoices", type_="unique")
    op.drop_constraint("chk_invoice_refundable_nonneg", "invoices", type_="check")
    op.drop_column("invoices", "refundable_chf")
