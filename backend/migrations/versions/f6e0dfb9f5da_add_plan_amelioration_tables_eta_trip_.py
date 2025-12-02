"""add_plan_amelioration_tables_eta_trip_delay_archive

Revision ID: f6e0dfb9f5da
Revises: merge_68116559b15d_24bbcb82c891
Create Date: 2025-12-02 17:51:35.890428

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "f6e0dfb9f5da"
down_revision = "24bbcb82c891"  # Basé sur la révision actuelle
branch_labels = None
depends_on = None


def upgrade():
    # 1. Table eta_accuracy_log (Section 3.2.3)
    op.create_table(
        "eta_accuracy_log",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=True),
        sa.Column("assignment_id", sa.Integer(), nullable=True),
        sa.Column("predicted_eta_seconds", sa.Integer(), nullable=False),
        sa.Column("actual_duration_seconds", sa.Integer(), nullable=True),
        sa.Column("error_seconds", sa.Integer(), nullable=True),
        sa.Column("origin_lat", sa.Float(), nullable=False),
        sa.Column("origin_lon", sa.Float(), nullable=False),
        sa.Column("dest_lat", sa.Float(), nullable=False),
        sa.Column("dest_lon", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column("ml_confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"]),
        sa.ForeignKeyConstraint(["assignment_id"], ["assignment.id"]),
    )
    op.create_index(
        "ix_eta_accuracy_log_booking_id", "eta_accuracy_log", ["booking_id"]
    )
    op.create_index(
        "ix_eta_accuracy_log_assignment_id", "eta_accuracy_log", ["assignment_id"]
    )
    op.create_index(
        "ix_eta_accuracy_log_created_at", "eta_accuracy_log", ["created_at"]
    )

    # 2. Table trip_tracking (Section 3.3.3)
    op.create_table(
        "trip_tracking",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("assignment_id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("speed", sa.Float(), nullable=True),
        sa.Column("heading", sa.Float(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["assignment_id"], ["assignment.id"]),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"]),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"]),
    )
    op.create_index(
        "ix_trip_tracking_assignment_timestamp",
        "trip_tracking",
        ["assignment_id", "timestamp"],
    )
    op.create_index("ix_trip_tracking_booking_id", "trip_tracking", ["booking_id"])
    op.create_index("ix_trip_tracking_driver_id", "trip_tracking", ["driver_id"])
    op.create_index("ix_trip_tracking_timestamp", "trip_tracking", ["timestamp"])

    # 3. Table delay_events (Section 3.5.1)
    op.create_table(
        "delay_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("assignment_id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("delay_minutes", sa.Integer(), nullable=False),
        sa.Column("severity", sa.String(length=20), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cause", sa.String(length=100), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["assignment_id"], ["assignment.id"]),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"]),
    )
    op.create_index("ix_delay_events_assignment_id", "delay_events", ["assignment_id"])
    op.create_index("ix_delay_events_booking_id", "delay_events", ["booking_id"])
    op.create_index("ix_delay_events_detected_at", "delay_events", ["detected_at"])
    op.create_index("ix_delay_events_severity", "delay_events", ["severity"])
    op.create_index("ix_delay_events_resolved_at", "delay_events", ["resolved_at"])

    # 4. Table trip_tracking_archive (Section 3.5.2) - Partitionnée par mois
    # Note: Le partitioning sera géré par le code Python (TripTrackingArchive.ensure_partition_for_month)
    op.create_table(
        "trip_tracking_archive",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("assignment_id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False),
        sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("speed", sa.Float(), nullable=True),
        sa.Column("heading", sa.Float(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["assignment_id"], ["assignment.id"]),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"]),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"]),
    )
    op.create_index(
        "ix_trip_tracking_archive_assignment_timestamp",
        "trip_tracking_archive",
        ["assignment_id", "timestamp"],
    )
    op.create_index(
        "ix_trip_tracking_archive_booking_id", "trip_tracking_archive", ["booking_id"]
    )
    op.create_index(
        "ix_trip_tracking_archive_driver_id", "trip_tracking_archive", ["driver_id"]
    )
    op.create_index(
        "ix_trip_tracking_archive_timestamp", "trip_tracking_archive", ["timestamp"]
    )


def downgrade():
    # Supprimer dans l'ordre inverse
    op.drop_index(
        "ix_trip_tracking_archive_timestamp", table_name="trip_tracking_archive"
    )
    op.drop_index(
        "ix_trip_tracking_archive_driver_id", table_name="trip_tracking_archive"
    )
    op.drop_index(
        "ix_trip_tracking_archive_booking_id", table_name="trip_tracking_archive"
    )
    op.drop_index(
        "ix_trip_tracking_archive_assignment_timestamp",
        table_name="trip_tracking_archive",
    )
    op.drop_table("trip_tracking_archive")

    op.drop_index("ix_delay_events_resolved_at", table_name="delay_events")
    op.drop_index("ix_delay_events_severity", table_name="delay_events")
    op.drop_index("ix_delay_events_detected_at", table_name="delay_events")
    op.drop_index("ix_delay_events_booking_id", table_name="delay_events")
    op.drop_index("ix_delay_events_assignment_id", table_name="delay_events")
    op.drop_table("delay_events")

    op.drop_index("ix_trip_tracking_timestamp", table_name="trip_tracking")
    op.drop_index("ix_trip_tracking_driver_id", table_name="trip_tracking")
    op.drop_index("ix_trip_tracking_booking_id", table_name="trip_tracking")
    op.drop_index("ix_trip_tracking_assignment_timestamp", table_name="trip_tracking")
    op.drop_table("trip_tracking")

    op.drop_index("ix_eta_accuracy_log_created_at", table_name="eta_accuracy_log")
    op.drop_index("ix_eta_accuracy_log_assignment_id", table_name="eta_accuracy_log")
    op.drop_index("ix_eta_accuracy_log_booking_id", table_name="eta_accuracy_log")
    op.drop_table("eta_accuracy_log")
