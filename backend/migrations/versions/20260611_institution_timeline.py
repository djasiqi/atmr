"""Institution timeline, change requests, multi-stop legs.

Revision ID: 20260611_institution_timeline
Revises: 20260610_native_start_diag
Create Date: 2026-06-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260611_institution_timeline"
down_revision = "20260610_native_start_diag"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "transport_timeline_events",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("transport_request_id", sa.Integer(), nullable=True),
        sa.Column("booking_id", sa.Integer(), nullable=True),
        sa.Column("institution_id", sa.Integer(), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor_type", sa.String(length=32), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column("driver_id", sa.Integer(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "payload_version", sa.SmallInteger(), server_default="1", nullable=False
        ),
        sa.Column("correlation_id", sa.String(length=100), nullable=True),
        sa.Column("source_event_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["institution_id"], ["institutions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["source_event_id"], ["transport_timeline_events.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["transport_request_id"], ["transport_requests.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_tte_request_created",
        "transport_timeline_events",
        ["transport_request_id", "created_at"],
    )
    op.create_index(
        "ix_tte_booking_created",
        "transport_timeline_events",
        ["booking_id", "created_at"],
    )
    op.create_index(
        "ix_tte_institution_created",
        "transport_timeline_events",
        ["institution_id", "created_at"],
    )
    op.create_index(
        "ix_tte_source_event", "transport_timeline_events", ["source_event_id"]
    )
    op.create_index(
        "ix_tte_correlation", "transport_timeline_events", ["correlation_id"]
    )
    op.create_index("ix_tte_event_type", "transport_timeline_events", ["event_type"])

    op.create_table(
        "booking_change_requests",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("transport_request_id", sa.Integer(), nullable=True),
        sa.Column("institution_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Integer(), server_default="1", nullable=False),
        sa.Column(
            "proposed_patch", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column(
            "before_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "after_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "changed_fields", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("requested_by_user_id", sa.Integer(), nullable=True),
        sa.Column("requested_by_role", sa.String(length=64), nullable=True),
        sa.Column("responded_by_user_id", sa.Integer(), nullable=True),
        sa.Column("responded_by_role", sa.String(length=64), nullable=True),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["institution_id"], ["institutions.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["requested_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["responded_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["transport_request_id"], ["transport_requests.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_bcr_booking_status", "booking_change_requests", ["booking_id", "status"]
    )
    op.create_index(
        "ix_bcr_institution_created",
        "booking_change_requests",
        ["institution_id", "created_at"],
    )

    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("active_change_request_id", sa.BigInteger(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("route_group_id", sa.String(length=36), nullable=True)
        )
        batch_op.add_column(
            sa.Column("route_sequence_number", sa.Integer(), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_booking_active_change_request",
            "booking_change_requests",
            ["active_change_request_id"],
            ["id"],
            ondelete="SET NULL",
        )

    with op.batch_alter_table("transport_requests", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "multi_stop", sa.Boolean(), server_default="false", nullable=False
            )
        )
        batch_op.add_column(
            sa.Column(
                "return_to_institution",
                sa.Boolean(),
                server_default="false",
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column("route_group_id", sa.String(length=36), nullable=True)
        )

    op.create_table(
        "transport_request_legs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("transport_request_id", sa.Integer(), nullable=False),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("route_sequence_number", sa.Integer(), nullable=False),
        sa.Column("pickup_location", sa.String(length=255), nullable=False),
        sa.Column("pickup_lat", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("pickup_lng", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("dropoff_location", sa.String(length=255), nullable=False),
        sa.Column("dropoff_lat", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("dropoff_lng", sa.Numeric(precision=10, scale=7), nullable=True),
        sa.Column("scheduled_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("booking_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["transport_request_id"], ["transport_requests.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_transport_request_leg_sequence",
        "transport_request_legs",
        ["transport_request_id", "sequence_index"],
        unique=True,
    )


def downgrade():
    op.drop_index(
        "uq_transport_request_leg_sequence", table_name="transport_request_legs"
    )
    op.drop_table("transport_request_legs")

    with op.batch_alter_table("transport_requests", schema=None) as batch_op:
        batch_op.drop_column("route_group_id")
        batch_op.drop_column("return_to_institution")
        batch_op.drop_column("multi_stop")

    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.drop_constraint("fk_booking_active_change_request", type_="foreignkey")
        batch_op.drop_column("route_sequence_number")
        batch_op.drop_column("route_group_id")
        batch_op.drop_column("active_change_request_id")

    op.drop_index("ix_bcr_institution_created", table_name="booking_change_requests")
    op.drop_index("ix_bcr_booking_status", table_name="booking_change_requests")
    op.drop_table("booking_change_requests")

    op.drop_index("ix_tte_event_type", table_name="transport_timeline_events")
    op.drop_index("ix_tte_correlation", table_name="transport_timeline_events")
    op.drop_index("ix_tte_source_event", table_name="transport_timeline_events")
    op.drop_index("ix_tte_institution_created", table_name="transport_timeline_events")
    op.drop_index("ix_tte_booking_created", table_name="transport_timeline_events")
    op.drop_index("ix_tte_request_created", table_name="transport_timeline_events")
    op.drop_table("transport_timeline_events")
