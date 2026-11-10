"""politique de reglement sur la reservation

Revision ID: b3e7a91c04d2
Revises: e6b4c91a07d3
Create Date: 2026-10-26 01:20:00.000000

Migration limitée aux deux tables de règlement. L'autogénération globale
reprendrait d'autres écarts de modèles du worktree. atmr_test est déjà
sur e6b4c91a07d3 : seule cette révision doit être appliquée, et seulement
sur cette base.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "b3e7a91c04d2"
down_revision = "e6b4c91a07d3"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "booking_settlement_policy",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("route_group_id", sa.String(length=36), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("modality", sa.String(length=40), nullable=False),
        sa.Column("authorized_modes", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("payment_scope", sa.String(length=20), nullable=True),
        sa.Column("collection_moment", sa.String(length=20), nullable=True),
        sa.Column("amount_chf", sa.Numeric(10, 2), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("decision_origin", sa.String(length=20), nullable=False),
        sa.Column("policy_version", sa.String(length=40), nullable=False),
        sa.Column(
            "financial_responsibility_locked",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by_user_id", sa.Integer(), nullable=True),
        sa.Column("invoice_signature_text", sa.Text(), nullable=True),
        sa.Column("invoice_signature_text_version", sa.String(length=40), nullable=True),
        sa.Column("validated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("validated_by_user_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("supersedes_revision", sa.Integer(), nullable=True),
        sa.Column(
            "round_trip_collection_approved",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "separate_leg_collection_approved",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "modality IN ('on_site', 'patient_invoice_after', "
            "'established_account', 'prepaid', 'institution_account', 'third_party')",
            name="ck_settlement_policy_modality",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'superseded')",
            name="ck_settlement_policy_status",
        ),
        sa.CheckConstraint("currency = 'CHF'", name="ck_settlement_policy_currency"),
        sa.CheckConstraint(
            "decision_origin IN ('client', 'institution', 'company')",
            name="ck_settlement_policy_origin",
        ),
        sa.CheckConstraint(
            "payment_scope IS NULL OR payment_scope IN ('booking', 'group')",
            name="ck_settlement_policy_scope",
        ),
        sa.CheckConstraint(
            "collection_moment IS NULL OR collection_moment IN ('outbound', 'return')",
            name="ck_settlement_policy_moment",
        ),
        sa.CheckConstraint(
            "(financial_responsibility_locked = false) OR "
            "(locked_at IS NOT NULL AND locked_by_user_id IS NOT NULL)",
            name="ck_settlement_policy_lock",
        ),
        sa.ForeignKeyConstraint(
            ["booking_id"], ["booking.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["locked_by_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["validated_by_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_booking_settlement_policy_revision",
        "booking_settlement_policy",
        ["booking_id", "revision"],
        unique=True,
    )
    op.create_index(
        "uq_booking_settlement_policy_active",
        "booking_settlement_policy",
        ["booking_id"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "company_settlement_capability",
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column(
            "tap_to_pay_enabled",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "cash_enabled",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "direct_patient_invoice_enabled",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_by_user_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["updated_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("company_id"),
    )


def downgrade():
    op.drop_table("company_settlement_capability")
    op.drop_index(
        "uq_booking_settlement_policy_active",
        table_name="booking_settlement_policy",
    )
    op.drop_index(
        "uq_booking_settlement_policy_revision",
        table_name="booking_settlement_policy",
    )
    op.drop_table("booking_settlement_policy")
