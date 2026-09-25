"""portal_double_validation_7b

Revision ID: 3041a7d1f098
Revises: 773df2373b29
Create Date: 2026-09-24 10:44:24.203712

Migration additive uniquement — aucun backfill de preuves historiques.
"""

from alembic import op
import sqlalchemy as sa


revision = "3041a7d1f098"
down_revision = "773df2373b29"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "company_portal_cancellation_policy",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("body_text", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False),
        sa.Column(
            "effective_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "company_id",
            "version",
            name="uq_company_portal_cancellation_policy_version",
        ),
    )
    op.create_index(
        "uq_company_portal_cancellation_policy_current",
        "company_portal_cancellation_policy",
        ["company_id"],
        unique=True,
        postgresql_where=sa.text("is_current IS TRUE"),
    )
    op.create_table(
        "portal_carrier_offer",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("company_name_snapshot", sa.String(length=255), nullable=False),
        sa.Column("offered_amount", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("cancellation_policy_id", sa.Integer(), nullable=False),
        sa.Column("cancellation_policy_version", sa.String(length=32), nullable=False),
        sa.Column("cancellation_policy_snapshot", sa.Text(), nullable=False),
        sa.Column("cancellation_policy_hash", sa.String(length=64), nullable=False),
        sa.Column("offer_content_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "offered_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('offered', 'confirmed', 'stale', 'withdrawn')",
            name="ck_portal_carrier_offer_status",
        ),
        sa.CheckConstraint(
            "offered_amount > 0",
            name="ck_portal_carrier_offer_amount_positive",
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["cancellation_policy_id"],
            ["company_portal_cancellation_policy.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "offer_content_hash", name="uq_portal_carrier_offer_content_hash"
        ),
    )
    op.create_index(
        op.f("ix_portal_carrier_offer_booking_id"),
        "portal_carrier_offer",
        ["booking_id"],
        unique=False,
    )
    op.create_index(
        "uq_portal_carrier_offer_active",
        "portal_carrier_offer",
        ["booking_id"],
        unique=True,
        postgresql_where=sa.text("status = 'offered'"),
    )
    op.create_table(
        "portal_client_transport_confirmation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("carrier_offer_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("company_name_snapshot", sa.String(length=255), nullable=False),
        sa.Column(
            "contractual_amount", sa.Numeric(precision=10, scale=2), nullable=False
        ),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("cancellation_policy_version", sa.String(length=32), nullable=False),
        sa.Column("cancellation_policy_snapshot", sa.Text(), nullable=False),
        sa.Column("cancellation_policy_hash", sa.String(length=64), nullable=False),
        sa.Column("carrier_offer_hash", sa.String(length=64), nullable=False),
        sa.Column("confirmed_by_user_id", sa.Integer(), nullable=False),
        sa.Column(
            "confirmed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "contractual_amount > 0",
            name="ck_portal_client_transport_confirmation_amount",
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["carrier_offer_id"],
            ["portal_carrier_offer.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["confirmed_by_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "booking_id", name="uq_portal_client_transport_confirmation_booking"
        ),
        sa.UniqueConstraint(
            "carrier_offer_id",
            name="uq_portal_client_transport_confirmation_offer",
        ),
    )
    op.add_column(
        "booking",
        sa.Column("portal_contract_flow", sa.String(length=32), nullable=True),
    )
    op.add_column(
        "client_booking_contract_event",
        sa.Column("maximum_accepted_amount_snapshot", sa.Float(), nullable=True),
    )


def downgrade():
    op.drop_column("client_booking_contract_event", "maximum_accepted_amount_snapshot")
    op.drop_column("booking", "portal_contract_flow")
    op.drop_table("portal_client_transport_confirmation")
    op.drop_index(
        "uq_portal_carrier_offer_active",
        table_name="portal_carrier_offer",
        postgresql_where=sa.text("status = 'offered'"),
    )
    op.drop_index(
        op.f("ix_portal_carrier_offer_booking_id"), table_name="portal_carrier_offer"
    )
    op.drop_table("portal_carrier_offer")
    op.drop_index(
        "uq_company_portal_cancellation_policy_current",
        table_name="company_portal_cancellation_policy",
        postgresql_where=sa.text("is_current IS TRUE"),
    )
    op.drop_table("company_portal_cancellation_policy")
