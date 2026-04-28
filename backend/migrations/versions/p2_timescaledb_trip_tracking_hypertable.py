"""P0-1 Phase 4: Convert trip_tracking to TimescaleDB hypertable.

Converts the trip_tracking table to a TimescaleDB hypertable partitioned by
timestamp, then adds compression (after 7 days) and retention (drop after
90 days) policies.

Requires TimescaleDB extension. The migration is safe to run on a live
database; create_hypertable uses migrate_data=TRUE to handle existing rows.

Revision ID: p2_timescaledb_tracking
Revises: p1_assignment_indexes
Create Date: 2026-04-19
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "p2_timescaledb_tracking"
down_revision = "p1_assignment_indexes"
branch_labels = None
depends_on = None


def _timescaledb_available(conn) -> bool:
    """Check TimescaleDB extension is installed before applying policies."""
    result = conn.execute(
        sa.text(
            "SELECT COUNT(*) FROM pg_extension WHERE extname = 'timescaledb'"
        )
    )
    return result.scalar() > 0


def upgrade() -> None:
    conn = op.get_bind()

    if not _timescaledb_available(conn):
        # Graceful degradation: skip hypertable creation if extension absent.
        # The table remains a plain PG table — performance is degraded but the
        # app remains functional. The DBA must install TimescaleDB and re-run.
        print(
            "[p2_timescaledb_tracking] SKIP: TimescaleDB extension not found. "
            "Install timescaledb and re-run this migration."
        )
        return

    # 1. Drop existing indexes that conflict with hypertable chunk constraints.
    #    TimescaleDB requires the partitioning column (timestamp) to be part of
    #    any unique index. These are plain non-unique indexes — safe to drop and
    #    recreate after conversion.
    op.execute(
        sa.text("DROP INDEX IF EXISTS ix_trip_tracking_assignment_timestamp")
    )
    op.execute(
        sa.text("DROP INDEX IF EXISTS ix_trip_tracking_timestamp")
    )

    # 2. Convert to hypertable.
    #    chunk_time_interval = 1 day: each chunk covers one day of tracking
    #    data, giving predictable chunk sizes (~10 GB/day at 100k drivers).
    #    migrate_data = TRUE: moves existing rows into the first chunk.
    conn.execute(
        sa.text(
            """
            SELECT create_hypertable(
                'trip_tracking',
                'timestamp',
                chunk_time_interval => INTERVAL '1 day',
                migrate_data => TRUE,
                if_not_exists => TRUE
            )
            """
        )
    )

    # 3. Recreate indexes as chunk-aware indexes (TimescaleDB handles fanout).
    #    Include timestamp in the composite index so chunk pruning applies.
    op.create_index(
        "ix_trip_tracking_assignment_timestamp",
        "trip_tracking",
        ["assignment_id", "timestamp"],
    )
    op.create_index(
        "ix_trip_tracking_driver_timestamp",
        "trip_tracking",
        ["driver_id", "timestamp"],
    )

    # 4. Compression policy: compress chunks older than 7 days.
    #    Segment by driver_id so queries for a single driver decompress only
    #    their own segments (columnar storage per driver).
    conn.execute(
        sa.text(
            """
            ALTER TABLE trip_tracking SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'driver_id',
                timescaledb.compress_orderby = 'timestamp DESC'
            )
            """
        )
    )
    conn.execute(
        sa.text(
            """
            SELECT add_compression_policy(
                'trip_tracking',
                INTERVAL '7 days',
                if_not_exists => TRUE
            )
            """
        )
    )

    # 5. Retention policy: drop chunks older than 90 days.
    #    Before production, confirm with the product team that 90 days covers
    #    all regulatory and analytics requirements. Extend to 365 days if
    #    billing or dispute resolution requires longer history.
    conn.execute(
        sa.text(
            """
            SELECT add_retention_policy(
                'trip_tracking',
                INTERVAL '90 days',
                if_not_exists => TRUE
            )
            """
        )
    )


def downgrade() -> None:
    conn = op.get_bind()

    if not _timescaledb_available(conn):
        return

    # Remove policies first (idempotent — no error if already absent).
    conn.execute(
        sa.text(
            "SELECT remove_retention_policy('trip_tracking', if_not_exists => TRUE)"
        )
    )
    conn.execute(
        sa.text(
            "SELECT remove_compression_policy('trip_tracking', if_not_exists => TRUE)"
        )
    )

    # Decompress all chunks before reverting (required before drop_chunks /
    # revert hypertable; decompression may take minutes on large datasets).
    conn.execute(
        sa.text(
            "SELECT decompress_chunk(c) FROM show_chunks('trip_tracking') c"
        )
    )

    # Revert to plain table.
    # NOTE: TimescaleDB does not support in-place un-hypertabling. The standard
    # approach is to recreate as a regular table and copy data back — which is
    # destructive for large datasets. We document this here but do NOT execute
    # it automatically to avoid accidental data loss.
    # To revert manually:
    #   CREATE TABLE trip_tracking_plain AS SELECT * FROM trip_tracking;
    #   DROP TABLE trip_tracking CASCADE;
    #   ALTER TABLE trip_tracking_plain RENAME TO trip_tracking;
    #   (re-add constraints and indexes from the original migration)
    print(
        "[p2_timescaledb_tracking] DOWNGRADE: Hypertable cannot be reverted "
        "automatically. See migration source for manual rollback procedure."
    )

    # Restore original plain indexes if somehow available.
    op.execute(
        sa.text("DROP INDEX IF EXISTS ix_trip_tracking_driver_timestamp")
    )
    op.execute(
        sa.text("DROP INDEX IF EXISTS ix_trip_tracking_assignment_timestamp")
    )
    op.create_index(
        "ix_trip_tracking_assignment_timestamp",
        "trip_tracking",
        ["assignment_id", "timestamp"],
    )
    op.create_index(
        "ix_trip_tracking_timestamp",
        "trip_tracking",
        ["timestamp"],
    )
