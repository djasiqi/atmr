"""Booking change audit tables + optimistic edit_version.

Revision ID: 20260526_booking_change_audit
Revises: 20260525_req_ext_ref_optional
Create Date: 2026-05-26 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "20260526_booking_change_audit"
down_revision = "20260525_req_ext_ref_optional"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "edit_version",
                sa.Integer(),
                nullable=False,
                server_default="1",
            )
        )

    op.create_table(
        "booking_change_events",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("transport_request_id", sa.Integer(), nullable=True),
        sa.Column("institution_id", sa.Integer(), nullable=True),
        sa.Column("booking_version", sa.Integer(), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("actor_role", sa.String(length=64), nullable=True),
        sa.Column("actor_type", sa.String(length=32), nullable=False),
        sa.Column("actor_display_name", sa.String(length=200), nullable=True),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column("change_class", sa.String(length=16), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column(
            "before_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "after_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "changed_fields",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("change_scope", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column(
            "operational_impact",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("financial_actor_role", sa.String(length=32), nullable=True),
        sa.Column("billing_change_reason_code", sa.String(length=64), nullable=True),
        sa.Column(
            "ack_required",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("correlation_id", sa.String(length=100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["institution_id"], ["institutions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["transport_request_id"],
            ["transport_requests.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_bce_booking_created",
        "booking_change_events",
        ["booking_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_bce_severity_ack_created",
        "booking_change_events",
        ["severity", "ack_required", "created_at"],
        unique=False,
    )
    op.create_index(
        "ix_bce_correlation",
        "booking_change_events",
        ["correlation_id"],
        unique=False,
    )
    op.create_index(
        "ix_bce_institution_created",
        "booking_change_events",
        ["institution_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_booking_change_events_booking_id"),
        "booking_change_events",
        ["booking_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_booking_change_events_institution_id"),
        "booking_change_events",
        ["institution_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_booking_change_events_transport_request_id"),
        "booking_change_events",
        ["transport_request_id"],
        unique=False,
    )

    op.create_table(
        "booking_change_acknowledgements",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("event_id", sa.BigInteger(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("actor_type", sa.String(length=32), nullable=False),
        sa.Column(
            "ack_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ack_channel", sa.String(length=32), nullable=True),
        sa.Column(
            "ack_metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["event_id"], ["booking_change_events.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_bce_ack_event_user_actor",
        "booking_change_acknowledgements",
        ["event_id", "user_id", "actor_type"],
        unique=True,
    )
    op.create_index(
        "ix_bce_ack_event",
        "booking_change_acknowledgements",
        ["event_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_booking_change_acknowledgements_user_id"),
        "booking_change_acknowledgements",
        ["user_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        op.f("ix_booking_change_acknowledgements_user_id"),
        table_name="booking_change_acknowledgements",
    )
    op.drop_index("ix_bce_ack_event", table_name="booking_change_acknowledgements")
    op.drop_index(
        "uq_bce_ack_event_user_actor", table_name="booking_change_acknowledgements"
    )
    op.drop_table("booking_change_acknowledgements")

    op.drop_index(
        op.f("ix_booking_change_events_transport_request_id"),
        table_name="booking_change_events",
    )
    op.drop_index(
        op.f("ix_booking_change_events_institution_id"),
        table_name="booking_change_events",
    )
    op.drop_index(
        op.f("ix_booking_change_events_booking_id"), table_name="booking_change_events"
    )
    op.drop_index("ix_bce_institution_created", table_name="booking_change_events")
    op.drop_index("ix_bce_correlation", table_name="booking_change_events")
    op.drop_index("ix_bce_severity_ack_created", table_name="booking_change_events")
    op.drop_index("ix_bce_booking_created", table_name="booking_change_events")
    op.drop_table("booking_change_events")

    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.drop_column("edit_version")
