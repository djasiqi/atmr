"""Fondation GPS Kafka-first v5 : sessions, ledger étendu, events, outbox, enrichments.

Revision ID: 20260727_gps_v5
Revises: 20260726_col_comments
Create Date: 2026-07-27

Migration manuelle pour PARTITION BY RANGE (autogenerate insuffisant — plan Annexe A).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260727_gps_v5"
down_revision = "20260726_col_comments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE SEQUENCE IF NOT EXISTS tracking_session_generation_seq AS BIGINT START WITH 1"
    )

    op.create_table(
        "tracking_sessions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("tracking_session_id", sa.String(length=128), nullable=False),
        sa.Column("session_generation", sa.BigInteger(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("final_sequence_id", sa.BigInteger(), nullable=True),
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
        sa.ForeignKeyConstraint(["company_id"], ["company.id"]),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "driver_id",
            "tracking_session_id",
            name="uq_tracking_sessions_driver_session",
        ),
        sa.UniqueConstraint(
            "driver_id",
            "session_generation",
            name="uq_tracking_sessions_driver_generation",
        ),
    )
    op.create_index(
        "ix_tracking_sessions_status",
        "tracking_sessions",
        ["driver_id", "status"],
    )
    op.create_index(
        op.f("ix_tracking_sessions_driver_id"),
        "tracking_sessions",
        ["driver_id"],
    )
    op.create_index(
        op.f("ix_tracking_sessions_company_id"),
        "tracking_sessions",
        ["company_id"],
    )

    op.create_table(
        "tracking_session_state",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("tracking_session_id", sa.String(length=128), nullable=False),
        sa.Column("session_generation", sa.BigInteger(), nullable=False),
        sa.Column(
            "contiguous_persisted_through",
            sa.BigInteger(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "max_seen_sequence", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "driver_id",
            "tracking_session_id",
            name="uq_tracking_session_state",
        ),
    )
    op.create_index(
        op.f("ix_tracking_session_state_driver_id"),
        "tracking_session_state",
        ["driver_id"],
    )

    op.create_table(
        "tracking_sequence_gaps",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("tracking_session_id", sa.String(length=128), nullable=False),
        sa.Column("sequence_from", sa.BigInteger(), nullable=False),
        sa.Column("sequence_to", sa.BigInteger(), nullable=False),
        sa.Column(
            "detected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_tracking_gaps_session",
        "tracking_sequence_gaps",
        ["driver_id", "tracking_session_id", "resolved_at"],
    )

    op.create_table(
        "tracking_event_outbox",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("location_event_id", sa.String(length=64), nullable=False),
        sa.Column(
            "session_generation", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column("sequence_id", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("event_id", name="uq_tracking_outbox_event_id"),
    )
    op.create_index(
        "ix_tracking_outbox_pending",
        "tracking_event_outbox",
        ["driver_id", "published_at", "session_generation", "sequence_id"],
    )
    op.create_index(
        op.f("ix_tracking_event_outbox_driver_id"),
        "tracking_event_outbox",
        ["driver_id"],
    )

    # Table parent partitionnée — SQL brut (Annexe A)
    op.execute(
        """
        CREATE TABLE driver_location_events (
            id BIGSERIAL,
            driver_id INTEGER NOT NULL,
            company_id INTEGER NOT NULL,
            location_event_id VARCHAR(64) NOT NULL,
            tracking_session_id VARCHAR(128) NOT NULL,
            session_generation BIGINT NOT NULL,
            sequence_id BIGINT NOT NULL,
            recorded_at TIMESTAMPTZ NOT NULL,
            raw_latitude DOUBLE PRECISION NOT NULL,
            raw_longitude DOUBLE PRECISION NOT NULL,
            accuracy_m DOUBLE PRECISION,
            speed_mps DOUBLE PRECISION,
            heading DOUBLE PRECISION,
            location_mode VARCHAR(32) NOT NULL,
            mission_id INTEGER,
            source VARCHAR(32) NOT NULL,
            event_payload_hash VARCHAR(64) NOT NULL,
            payload_schema_version VARCHAR(32) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (id, recorded_at)
        ) PARTITION BY RANGE (recorded_at)
        """
    )
    op.execute(
        """
        CREATE TABLE driver_location_events_2026_07
        PARTITION OF driver_location_events
        FOR VALUES FROM ('2026-07-01') TO ('2026-08-01')
        """
    )
    op.execute(
        """
        CREATE TABLE driver_location_events_2026_08
        PARTITION OF driver_location_events
        FOR VALUES FROM ('2026-08-01') TO ('2026-09-01')
        """
    )
    op.execute(
        """
        CREATE TABLE driver_location_events_default
        PARTITION OF driver_location_events DEFAULT
        """
    )
    op.execute(
        "CREATE INDEX ix_dle_driver_recorded ON driver_location_events (driver_id, recorded_at)"
    )
    op.execute(
        "CREATE INDEX ix_dle_driver_event ON driver_location_events (driver_id, location_event_id)"
    )

    op.create_table(
        "driver_location_enrichments",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("location_event_id", sa.String(length=64), nullable=False),
        sa.Column("enrichment_version", sa.Integer(), nullable=False),
        sa.Column("canonical_latitude", sa.Float(), nullable=False),
        sa.Column("canonical_longitude", sa.Float(), nullable=False),
        sa.Column("canonical_source", sa.String(length=32), nullable=False),
        sa.Column("processing_status", sa.String(length=16), nullable=False),
        sa.Column(
            "enriched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "driver_id",
            "location_event_id",
            "enrichment_version",
            name="uq_dle_enrichment_version",
        ),
    )

    # Projection driver ordonnée (Annexe A.9)
    op.add_column(
        "driver",
        sa.Column("last_location_event_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "driver",
        sa.Column("last_tracking_session_generation", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "driver",
        sa.Column("last_tracking_sequence_id", sa.BigInteger(), nullable=True),
    )

    # Colonnes ledger pour session/séquence + FK soft (après tracking_sessions)
    op.add_column(
        "tracking_ingest_events",
        sa.Column("tracking_session_id", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "tracking_ingest_events",
        sa.Column("sequence_id", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "tracking_ingest_events",
        sa.Column("session_generation", sa.BigInteger(), nullable=True),
    )
    op.create_index(
        "ix_tracking_ingest_session_seq",
        "tracking_ingest_events",
        ["driver_id", "tracking_session_id", "sequence_id"],
        unique=False,
    )
    # Unicité session/séquence (NULL-safe partiel)
    op.execute(
        """
        CREATE UNIQUE INDEX uq_tracking_ingest_session_sequence
        ON tracking_ingest_events (driver_id, tracking_session_id, sequence_id)
        WHERE tracking_session_id IS NOT NULL AND sequence_id IS NOT NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_tracking_ingest_session_sequence")
    op.drop_index("ix_tracking_ingest_session_seq", table_name="tracking_ingest_events")
    op.drop_column("tracking_ingest_events", "session_generation")
    op.drop_column("tracking_ingest_events", "sequence_id")
    op.drop_column("tracking_ingest_events", "tracking_session_id")

    op.drop_column("driver", "last_tracking_sequence_id")
    op.drop_column("driver", "last_tracking_session_generation")
    op.drop_column("driver", "last_location_event_id")

    op.drop_table("driver_location_enrichments")
    op.execute("DROP TABLE IF EXISTS driver_location_events CASCADE")

    op.drop_index(
        op.f("ix_tracking_event_outbox_driver_id"), table_name="tracking_event_outbox"
    )
    op.drop_index("ix_tracking_outbox_pending", table_name="tracking_event_outbox")
    op.drop_table("tracking_event_outbox")

    op.drop_index("ix_tracking_gaps_session", table_name="tracking_sequence_gaps")
    op.drop_table("tracking_sequence_gaps")

    op.drop_index(
        op.f("ix_tracking_session_state_driver_id"),
        table_name="tracking_session_state",
    )
    op.drop_table("tracking_session_state")

    op.drop_index(
        op.f("ix_tracking_sessions_company_id"), table_name="tracking_sessions"
    )
    op.drop_index(op.f("ix_tracking_sessions_driver_id"), table_name="tracking_sessions")
    op.drop_index("ix_tracking_sessions_status", table_name="tracking_sessions")
    op.drop_table("tracking_sessions")

    op.execute("DROP SEQUENCE IF EXISTS tracking_session_generation_seq")
