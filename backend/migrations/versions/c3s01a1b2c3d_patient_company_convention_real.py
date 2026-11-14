"""patient company convention real + events

Revision ID: c3s01a1b2c3d
Revises: c2r02a1b3c4d
Create Date: 2026-11-14 17:00:00.000000

Migration limitée au lot CONVENTION-SIGNATURE-REAL-01.
Ne modifie pas patient_company_convention_demo ni ck_convention_demo_not_legal.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "c3s01a1b2c3d"
down_revision = "c2r02a1b3c4d"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "patient_company_convention",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.Integer(), nullable=False),
        sa.Column("debtor_id", sa.Integer(), nullable=True),
        sa.Column("debtor_label", sa.String(length=160), nullable=False),
        sa.Column("invoice_recipient_id", sa.Integer(), nullable=True),
        sa.Column("invoice_recipient_label", sa.String(length=160), nullable=True),
        sa.Column("billing_conditions", sa.Text(), nullable=False),
        sa.Column("tariff_version", sa.String(length=64), nullable=False),
        sa.Column("service_scope", sa.Text(), nullable=False),
        sa.Column("document_version", sa.String(length=32), nullable=False),
        sa.Column(
            "version_number", sa.Integer(), server_default=sa.text("1"), nullable=False
        ),
        sa.Column(
            "active", sa.Boolean(), server_default=sa.text("true"), nullable=False
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column(
            "legally_effective",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "legal_activation_required",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "consent_acknowledged",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("signer_kind", sa.String(length=32), nullable=True),
        sa.Column("signer_name", sa.String(length=160), nullable=True),
        sa.Column(
            "signature_modality",
            sa.String(length=32),
            server_default=sa.text("'none'"),
            nullable=False,
        ),
        sa.Column("signed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("document_sha256", sa.String(length=64), nullable=True),
        sa.Column("archive_reference", sa.String(length=255), nullable=True),
        sa.Column("paper_scan_reference", sa.String(length=255), nullable=True),
        sa.Column(
            "signature_trace",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("valid_from", sa.Date(), nullable=True),
        sa.Column("valid_until", sa.Date(), nullable=True),
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
        sa.ForeignKeyConstraint(["client_id"], ["client.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
        sa.UniqueConstraint(
            "client_id", "company_id", "version_number", name="uq_pcc_version"
        ),
        sa.UniqueConstraint("company_id", "idempotency_key", name="uq_pcc_company_key"),
        sa.CheckConstraint(
            "status IN ("
            "'draft','awaiting_signature','signed_pending_validation',"
            "'applicable','expired','revoked','replaced')",
            name="ck_pcc_status",
        ),
        sa.CheckConstraint("version_number > 0", name="ck_pcc_version"),
        sa.CheckConstraint("char_length(idempotency_key) > 0", name="ck_pcc_key"),
        sa.CheckConstraint(
            "signature_modality IN ('none','on_screen_trace','paper_scan')",
            name="ck_pcc_modality",
        ),
        sa.CheckConstraint(
            "legally_effective = false OR status = 'applicable'",
            name="ck_pcc_legal_only_if_applicable",
        ),
        sa.CheckConstraint(
            "consent_acknowledged = false OR status IN ("
            "'signed_pending_validation','applicable','expired','revoked','replaced')",
            name="ck_pcc_consent_signed_only",
        ),
    )
    op.create_index("ix_pcc_company_id", "patient_company_convention", ["company_id"])
    op.create_index("ix_pcc_client_id", "patient_company_convention", ["client_id"])
    op.execute(
        "CREATE UNIQUE INDEX uq_pcc_active ON patient_company_convention "
        "(client_id, company_id) WHERE active"
    )

    op.create_table(
        "patient_company_convention_event",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("convention_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("actor_role", sa.String(length=32), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("detail", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["convention_id"],
            ["patient_company_convention.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "convention_id", "idempotency_key", name="uq_pcc_event_key"
        ),
        sa.CheckConstraint("char_length(idempotency_key) > 0", name="ck_pcc_event_key"),
    )
    op.create_index(
        "ix_pcc_event_convention",
        "patient_company_convention_event",
        ["convention_id"],
    )


def downgrade():
    op.drop_index(
        "ix_pcc_event_convention", table_name="patient_company_convention_event"
    )
    op.drop_table("patient_company_convention_event")
    op.execute("DROP INDEX IF EXISTS uq_pcc_active")
    op.drop_index("ix_pcc_client_id", table_name="patient_company_convention")
    op.drop_index("ix_pcc_company_id", table_name="patient_company_convention")
    op.drop_table("patient_company_convention")
