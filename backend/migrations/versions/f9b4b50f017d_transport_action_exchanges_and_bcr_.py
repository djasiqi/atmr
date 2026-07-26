"""transport_action_exchanges_and_bcr_fields

Revision ID: f9b4b50f017d
Revises: 19bfdea7a833
Create Date: 2026-07-21 14:26:11.831673

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "f9b4b50f017d"
down_revision = "19bfdea7a833"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "transport_action_exchanges",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("transport_action_id", sa.BigInteger(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("actor_type", sa.String(length=32), nullable=False),
        sa.Column("actor_id", sa.BigInteger(), nullable=True),
        sa.Column("decision_type", sa.String(length=32), nullable=False),
        sa.Column("values", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "commercial_terms", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_from", sa.String(length=32), nullable=True),
        sa.Column(
            "client_meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column(
            "decision_context_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["transport_action_id"],
            ["booking_change_requests.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "transport_action_id", "sequence", name="uq_tae_action_sequence"
        ),
    )
    op.create_index(
        "ix_tae_action_created",
        "transport_action_exchanges",
        ["transport_action_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_transport_action_exchanges_idempotency_key"),
        "transport_action_exchanges",
        ["idempotency_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_transport_action_exchanges_transport_action_id"),
        "transport_action_exchanges",
        ["transport_action_id"],
        unique=False,
    )

    op.add_column(
        "booking_change_requests",
        sa.Column("action_type", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column(
            "action_scope",
            sa.String(length=32),
            server_default="BOOKING",
            nullable=True,
        ),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column(
            "effect_status",
            sa.String(length=16),
            server_default="none",
            nullable=False,
        ),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column(
            "next_actor_type",
            sa.String(length=16),
            server_default="COMPANY",
            nullable=False,
        ),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column("active_exchange_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column("mission_version_at_request", sa.Integer(), nullable=True),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column("rejection_reason", sa.Text(), nullable=True),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column("billing_assessment_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column("viewed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column("claimed_by_user_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "booking_change_requests",
        sa.Column(
            "handling_status",
            sa.String(length=16),
            server_default="UNSEEN",
            nullable=True,
        ),
    )


def downgrade():
    op.drop_column("booking_change_requests", "handling_status")
    op.drop_column("booking_change_requests", "claimed_at")
    op.drop_column("booking_change_requests", "claimed_by_user_id")
    op.drop_column("booking_change_requests", "viewed_at")
    op.drop_column("booking_change_requests", "completed_at")
    op.drop_column("booking_change_requests", "billing_assessment_id")
    op.drop_column("booking_change_requests", "rejection_reason")
    op.drop_column("booking_change_requests", "mission_version_at_request")
    op.drop_column("booking_change_requests", "active_exchange_id")
    op.drop_column("booking_change_requests", "next_actor_type")
    op.drop_column("booking_change_requests", "effect_status")
    op.drop_column("booking_change_requests", "action_scope")
    op.drop_column("booking_change_requests", "action_type")

    op.drop_index(
        op.f("ix_transport_action_exchanges_transport_action_id"),
        table_name="transport_action_exchanges",
    )
    op.drop_index(
        op.f("ix_transport_action_exchanges_idempotency_key"),
        table_name="transport_action_exchanges",
    )
    op.drop_index("ix_tae_action_created", table_name="transport_action_exchanges")
    op.drop_table("transport_action_exchanges")
