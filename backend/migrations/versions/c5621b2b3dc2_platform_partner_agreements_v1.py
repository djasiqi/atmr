"""platform_partner_agreements_v1

Revision ID: c5621b2b3dc2
Revises: 8493bf35866f
Create Date: 2026-08-02 20:59:03.859239

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "c5621b2b3dc2"
down_revision = "8493bf35866f"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "platform_partner_agreement_sequence",
        sa.Column("year_month", sa.String(length=7), nullable=False),
        sa.Column("last_value", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("year_month"),
    )
    op.create_table(
        "platform_partner_agreement",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("billing_config_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column("reference", sa.String(length=64), nullable=False),
        sa.Column(
            "status", sa.String(length=16), server_default="draft", nullable=False
        ),
        sa.Column("generated_storage_key", sa.String(length=512), nullable=True),
        sa.Column("generated_sha256", sa.String(length=64), nullable=True),
        sa.Column("generated_size_bytes", sa.Integer(), nullable=True),
        sa.Column("generated_content_type", sa.String(length=128), nullable=True),
        sa.Column("signed_storage_key", sa.String(length=512), nullable=True),
        sa.Column("signed_sha256", sa.String(length=64), nullable=True),
        sa.Column("signed_size_bytes", sa.Integer(), nullable=True),
        sa.Column("signed_content_type", sa.String(length=128), nullable=True),
        sa.Column("signed_original_filename", sa.String(length=255), nullable=True),
        sa.Column(
            "parties_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "commercial_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "generation_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("signed_file_uploaded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("agreement_signed_on", sa.Date(), nullable=True),
        sa.Column("agreement_effective_from", sa.Date(), nullable=True),
        sa.Column("generated_by_user_id", sa.Integer(), nullable=True),
        sa.Column("sent_by_user_id", sa.Integer(), nullable=True),
        sa.Column("signed_uploaded_by_user_id", sa.Integer(), nullable=True),
        sa.Column("voided_by_user_id", sa.Integer(), nullable=True),
        sa.Column("void_reason", sa.Text(), nullable=True),
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
            "status IN ('draft', 'sent', 'signed', 'void')", name="ck_ppa_status"
        ),
        sa.CheckConstraint(
            "revision_number >= 1", name="ck_ppa_revision_number_positive"
        ),
        sa.ForeignKeyConstraint(
            ["billing_config_id"],
            ["company_platform_billing_config.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["generated_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["sent_by_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["signed_uploaded_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["voided_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "billing_config_id",
            "revision_number",
            name="uq_ppa_config_revision",
        ),
        sa.UniqueConstraint("reference", name="uq_ppa_reference"),
    )
    op.create_index(
        "ix_ppa_company_id",
        "platform_partner_agreement",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        "uq_ppa_active_per_config",
        "platform_partner_agreement",
        ["billing_config_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('draft', 'sent', 'signed')"),
    )

    op.add_column(
        "company",
        sa.Column(
            "legal_form",
            sa.String(length=32),
            nullable=True,
            comment="Forme juridique contractuelle (LegalForm)",
        ),
    )
    op.add_column(
        "company", sa.Column("signatory_name", sa.String(length=200), nullable=True)
    )
    op.add_column(
        "company", sa.Column("signatory_title", sa.String(length=120), nullable=True)
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column("free_license_max_months", sa.Integer(), nullable=True),
    )
    op.add_column(
        "company_platform_billing_config",
        sa.Column(
            "statement_dispute_days",
            sa.Integer(),
            server_default="10",
            nullable=True,
        ),
    )
    op.add_column(
        "platform_billing_creditor",
        sa.Column("legal_form", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "platform_billing_creditor",
        sa.Column("signatory_name", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "platform_billing_creditor",
        sa.Column("signatory_title", sa.String(length=120), nullable=True),
    )


def downgrade():
    op.drop_column("platform_billing_creditor", "signatory_title")
    op.drop_column("platform_billing_creditor", "signatory_name")
    op.drop_column("platform_billing_creditor", "legal_form")
    op.drop_column("company_platform_billing_config", "statement_dispute_days")
    op.drop_column("company_platform_billing_config", "free_license_max_months")
    op.drop_column("company", "signatory_title")
    op.drop_column("company", "signatory_name")
    op.drop_column("company", "legal_form")
    op.drop_index(
        "uq_ppa_active_per_config",
        table_name="platform_partner_agreement",
        postgresql_where=sa.text("status IN ('draft', 'sent', 'signed')"),
    )
    op.drop_index("ix_ppa_company_id", table_name="platform_partner_agreement")
    op.drop_table("platform_partner_agreement")
    op.drop_table("platform_partner_agreement_sequence")
