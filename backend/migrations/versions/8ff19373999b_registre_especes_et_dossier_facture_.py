"""registre especes et dossier facture patient

Revision ID: 8ff19373999b
Revises: a2167da2d6ae
Create Date: 2026-10-27 20:53:20.007620

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "8ff19373999b"
down_revision = "a2167da2d6ae"
branch_labels = None
depends_on = None


def upgrade():
    # Généré par Alembic, puis limité aux tables de préparation espèces
    # et facture patient. Le reste du diff reprenait un écart de modèles
    # déjà présent et ne fait pas partie de ce lot.
    op.create_table(
        "cash_collection",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("debtor_label", sa.String(length=160), nullable=False),
        sa.Column("amount_due_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("tendered_chf", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("change_chf", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("change_confirmed", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("net_chf", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("currency", sa.String(length=3), server_default=sa.text("'CHF'"), nullable=False),
        sa.Column("state", sa.String(length=32), server_default=sa.text("'prepared'"), nullable=False),
        sa.Column("disposition", sa.String(length=32), nullable=True),
        sa.Column("decision_version", sa.String(length=64), nullable=False),
        sa.Column("production_activation", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("legally_approved", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("currency = 'CHF'", name="ck_cash_collection_currency"),
        sa.CheckConstraint(
            "state IN ('prepared', 'received', 'returned', 'administrative', 'uncertain')",
            name="ck_cash_collection_state",
        ),
        sa.CheckConstraint("amount_due_chf > 0", name="ck_cash_collection_amount"),
        sa.CheckConstraint("legally_approved = false", name="ck_cash_collection_unapproved"),
        sa.CheckConstraint("production_activation = false", name="ck_cash_collection_inactive"),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("booking_id", name="uq_cash_collection_booking"),
    )
    op.create_table(
        "patient_billing_dossier",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("debtor_label", sa.String(length=160), nullable=False),
        sa.Column("amount_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("currency", sa.String(length=3), server_default=sa.text("'CHF'"), nullable=False),
        sa.Column("contractual_basis", sa.String(length=40), server_default=sa.text("'missing'"), nullable=False),
        sa.Column("document_version", sa.String(length=64), nullable=False),
        sa.Column("document_approved", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("legally_binding", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("real_signature_collected", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("invoice_issued", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("attestation_used_as_consent", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("production_activation", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("state", sa.String(length=32), server_default=sa.text("'preparing'"), nullable=False),
        sa.Column("decision_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint(
            "contractual_basis IN ('missing', 'prior_presented', "
            "'acceptance_unapproved', 'refused', 'representative', 'administrative')",
            name="ck_patient_billing_basis",
        ),
        sa.CheckConstraint("currency = 'CHF'", name="ck_patient_billing_currency"),
        sa.CheckConstraint(
            "state IN ('preparing', 'administrative', 'closed')",
            name="ck_patient_billing_state",
        ),
        sa.CheckConstraint("attestation_used_as_consent = false", name="ck_patient_billing_attestation"),
        sa.CheckConstraint("document_approved = false", name="ck_patient_billing_document"),
        sa.CheckConstraint("invoice_issued = false", name="ck_patient_billing_not_issued"),
        sa.CheckConstraint("legally_binding = false", name="ck_patient_billing_binding"),
        sa.CheckConstraint("production_activation = false", name="ck_patient_billing_inactive"),
        sa.CheckConstraint("real_signature_collected = false", name="ck_patient_billing_signature"),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("booking_id", name="uq_patient_billing_booking"),
    )
    op.create_table(
        "cash_collection_event",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("collection_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("action", sa.String(length=40), nullable=False),
        sa.Column("from_state", sa.String(length=32), nullable=False),
        sa.Column("to_state", sa.String(length=32), nullable=False),
        sa.Column("actor_role", sa.String(length=20), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["cash_collection.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_id", "idempotency_key", name="uq_cash_collection_event_key"),
    )
    op.create_index(
        "ix_cash_collection_event_collection",
        "cash_collection_event",
        ["collection_id", "id"],
        unique=False,
    )
    op.create_table(
        "cash_handover",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("collection_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("amount_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("state", sa.String(length=20), server_default=sa.text("'declared'"), nullable=False),
        sa.Column("declared_by_user_id", sa.Integer(), nullable=False),
        sa.Column("confirmed_by_user_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "(state = 'declared' AND confirmed_by_user_id IS NULL) OR "
            "(state = 'confirmed' AND confirmed_by_user_id IS NOT NULL)",
            name="ck_cash_handover_confirm",
        ),
        sa.CheckConstraint("state IN ('declared', 'confirmed')", name="ck_cash_handover_state"),
        sa.ForeignKeyConstraint(["collection_id"], ["cash_collection.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["confirmed_by_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["declared_by_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_id", name="uq_cash_handover_collection"),
    )
    op.create_table(
        "cash_receipt",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("collection_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("number", sa.String(length=40), nullable=False),
        sa.Column("issued", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("legal_status", sa.String(length=20), server_default=sa.text("'pending'"), nullable=False),
        sa.Column("specimen", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("duplicate_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("currency", sa.String(length=3), server_default=sa.text("'CHF'"), nullable=False),
        sa.Column("amount_due_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("tendered_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("change_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("net_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("currency = 'CHF'", name="ck_cash_receipt_currency"),
        sa.CheckConstraint("legal_status = 'pending'", name="ck_cash_receipt_legal_pending"),
        sa.CheckConstraint("issued = false", name="ck_cash_receipt_not_issued"),
        sa.CheckConstraint("specimen = true", name="ck_cash_receipt_specimen"),
        sa.ForeignKeyConstraint(["collection_id"], ["cash_collection.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("collection_id", name="uq_cash_receipt_collection"),
        sa.UniqueConstraint("company_id", "number", name="uq_cash_receipt_number"),
    )
    op.create_table(
        "patient_billing_event",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dossier_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("action", sa.String(length=40), nullable=False),
        sa.Column("from_state", sa.String(length=32), nullable=False),
        sa.Column("to_state", sa.String(length=32), nullable=False),
        sa.Column("actor_role", sa.String(length=20), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["dossier_id"], ["patient_billing_dossier.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("dossier_id", "idempotency_key", name="uq_patient_billing_event_key"),
    )
    op.create_index(
        "ix_patient_billing_event_dossier",
        "patient_billing_event",
        ["dossier_id", "id"],
        unique=False,
    )
    op.create_table(
        "cash_reconciliation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("handover_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("expected_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("counted_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("variance_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("author_user_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["author_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["handover_id"], ["cash_handover.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("handover_id", name="uq_cash_reconciliation_handover"),
    )


def downgrade():
    op.drop_table("cash_reconciliation")
    op.drop_index("ix_patient_billing_event_dossier", table_name="patient_billing_event")
    op.drop_table("patient_billing_event")
    op.drop_table("cash_receipt")
    op.drop_table("cash_handover")
    op.drop_index("ix_cash_collection_event_collection", table_name="cash_collection_event")
    op.drop_table("cash_collection_event")
    op.drop_table("patient_billing_dossier")
    op.drop_table("cash_collection")
