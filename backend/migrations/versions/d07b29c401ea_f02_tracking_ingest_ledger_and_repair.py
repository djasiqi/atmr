"""f02_tracking_ingest_ledger_and_repair

Revision ID: d07b29c401ea
Revises: 16f950e9a85f
Create Date: 2026-07-26 19:16:36.430532

"""

from alembic import op
import sqlalchemy as sa

revision = "d07b29c401ea"
down_revision = "16f950e9a85f"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "tracking_derived_repair_pending",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("location_event_id", sa.String(length=64), nullable=False),
        sa.Column("repair_kind", sa.String(length=32), nullable=False),
        sa.Column("target_recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("target_sequence_id", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "driver_id",
            "location_event_id",
            "repair_kind",
            name="uq_tracking_derived_repair",
        ),
    )
    op.create_index(
        op.f("ix_tracking_derived_repair_pending_driver_id"),
        "tracking_derived_repair_pending",
        ["driver_id"],
        unique=False,
    )
    op.create_table(
        "tracking_ingest_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("location_event_id", sa.String(length=64), nullable=False),
        sa.Column("event_payload_hash", sa.String(length=64), nullable=False),
        sa.Column("payload_schema_version", sa.String(length=32), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"]),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "driver_id",
            "location_event_id",
            name="uq_tracking_ingest_driver_event",
        ),
    )
    op.create_index(
        op.f("ix_tracking_ingest_events_company_id"),
        "tracking_ingest_events",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_tracking_ingest_events_driver_id"),
        "tracking_ingest_events",
        ["driver_id"],
        unique=False,
    )
    op.create_index(
        "ix_tracking_ingest_received_at",
        "tracking_ingest_events",
        ["received_at"],
        unique=False,
    )


def downgrade():
    op.drop_index("ix_tracking_ingest_received_at", table_name="tracking_ingest_events")
    op.drop_index(
        op.f("ix_tracking_ingest_events_driver_id"),
        table_name="tracking_ingest_events",
    )
    op.drop_index(
        op.f("ix_tracking_ingest_events_company_id"),
        table_name="tracking_ingest_events",
    )
    op.drop_table("tracking_ingest_events")
    op.drop_index(
        op.f("ix_tracking_derived_repair_pending_driver_id"),
        table_name="tracking_derived_repair_pending",
    )
    op.drop_table("tracking_derived_repair_pending")
