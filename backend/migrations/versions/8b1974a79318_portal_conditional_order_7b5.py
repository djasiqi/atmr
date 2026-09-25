"""portal_conditional_order_7b5

Revision ID: 8b1974a79318
Revises: b12686830f54
Create Date: 2026-09-25 00:45:50.369300

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "8b1974a79318"
down_revision = "b12686830f54"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "lirie_channel_cancellation_policy",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column(
            "body_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "version", name="uq_lirie_channel_cancellation_policy_version"
        ),
    )
    op.create_index(
        "uq_lirie_channel_cancellation_policy_current",
        "lirie_channel_cancellation_policy",
        ["is_current"],
        unique=True,
        postgresql_where=sa.text("is_current IS TRUE"),
    )

    op.create_table(
        "portal_client_conditional_order",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("flow_version", sa.String(length=32), nullable=False),
        sa.Column("client_ceiling", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column(
            "estimated_amount_snapshot",
            sa.Numeric(precision=10, scale=2),
            nullable=True,
        ),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column(
            "eligible_carriers_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("terms_of_service_version", sa.String(length=16), nullable=False),
        sa.Column("terms_of_service_hash", sa.String(length=64), nullable=False),
        sa.Column("transport_terms_version", sa.String(length=16), nullable=False),
        sa.Column("transport_terms_hash", sa.String(length=64), nullable=False),
        sa.Column("channel_policy_version", sa.String(length=32), nullable=False),
        sa.Column("channel_policy_hash", sa.String(length=64), nullable=False),
        sa.Column("channel_policy_snapshot", sa.Text(), nullable=False),
        sa.Column(
            "trip_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("debtor_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "client_ceiling > 0",
            name="ck_portal_client_conditional_order_ceiling",
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["debtor_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "booking_id", name="uq_portal_client_conditional_order_booking"
        ),
    )

    op.create_table(
        "portal_transport_contract_formed",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("carrier_legal_name", sa.String(length=255), nullable=False),
        sa.Column("carrier_quote", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("client_ceiling", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("flow_version", sa.String(length=32), nullable=False),
        sa.Column("company_policy_version", sa.String(length=32), nullable=False),
        sa.Column("company_policy_hash", sa.String(length=64), nullable=False),
        sa.Column("company_policy_snapshot", sa.Text(), nullable=False),
        sa.Column("channel_policy_version", sa.String(length=32), nullable=False),
        sa.Column("channel_policy_hash", sa.String(length=64), nullable=False),
        sa.Column("channel_policy_snapshot", sa.Text(), nullable=False),
        sa.Column("terms_of_service_version", sa.String(length=16), nullable=False),
        sa.Column("terms_of_service_hash", sa.String(length=64), nullable=False),
        sa.Column("transport_terms_version", sa.String(length=16), nullable=False),
        sa.Column("transport_terms_hash", sa.String(length=64), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "formed_at",
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
            "carrier_quote > 0",
            name="ck_portal_transport_contract_formed_quote",
        ),
        sa.CheckConstraint(
            "client_ceiling > 0",
            name="ck_portal_transport_contract_formed_ceiling",
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "booking_id", name="uq_portal_transport_contract_formed_booking"
        ),
    )


def downgrade():
    op.drop_table("portal_transport_contract_formed")
    op.drop_table("portal_client_conditional_order")
    op.drop_index(
        "uq_lirie_channel_cancellation_policy_current",
        table_name="lirie_channel_cancellation_policy",
        postgresql_where=sa.text("is_current IS TRUE"),
    )
    op.drop_table("lirie_channel_cancellation_policy")
