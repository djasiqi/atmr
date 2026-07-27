"""Registre de sessions tracking GPS (plan Kafka-first v5)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKeyConstraint, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from ext import db


class TrackingSession(db.Model):
    """Session de tracking chauffeur — autorité pour session_generation."""

    __tablename__ = "tracking_sessions"
    __table_args__ = (
        UniqueConstraint(
            "driver_id",
            "tracking_session_id",
            name="uq_tracking_sessions_driver_session",
        ),
        UniqueConstraint(
            "driver_id",
            "session_generation",
            name="uq_tracking_sessions_driver_generation",
        ),
        Index("ix_tracking_sessions_status", "driver_id", "status"),
    )

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    driver_id = db.Column(
        db.Integer, db.ForeignKey("driver.id"), nullable=False, index=True
    )
    company_id = db.Column(
        db.Integer, db.ForeignKey("company.id"), nullable=False, index=True
    )
    tracking_session_id = db.Column(db.String(128), nullable=False)
    session_generation = db.Column(db.BigInteger, nullable=False)
    status = db.Column(db.String(16), nullable=False, default="active")
    # active | superseded | closed | expired
    started_at = db.Column(db.DateTime(timezone=True), nullable=False)
    closed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    final_sequence_id = db.Column(db.BigInteger, nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=datetime.utcnow,
    )


class TrackingSessionState(db.Model):
    """Watermark contigu par session (source ACK persisted)."""

    __tablename__ = "tracking_session_state"
    __table_args__ = (
        UniqueConstraint(
            "driver_id",
            "tracking_session_id",
            name="uq_tracking_session_state",
        ),
    )

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    driver_id = db.Column(db.Integer, nullable=False, index=True)
    company_id = db.Column(db.Integer, nullable=False)
    tracking_session_id = db.Column(db.String(128), nullable=False)
    session_generation = db.Column(db.BigInteger, nullable=False)
    contiguous_persisted_through = db.Column(db.BigInteger, nullable=False, default=0)
    max_seen_sequence = db.Column(db.BigInteger, nullable=False, default=0)
    first_seen_at = db.Column(db.DateTime(timezone=True), nullable=False)
    last_seen_at = db.Column(db.DateTime(timezone=True), nullable=False)
    closed_at = db.Column(db.DateTime(timezone=True), nullable=True)


class TrackingSequenceGap(db.Model):
    """Gaps de séquence détectés côté serveur."""

    __tablename__ = "tracking_sequence_gaps"
    __table_args__ = (
        Index(
            "ix_tracking_gaps_session",
            "driver_id",
            "tracking_session_id",
            "resolved_at",
        ),
    )

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    driver_id = db.Column(db.Integer, nullable=False)
    tracking_session_id = db.Column(db.String(128), nullable=False)
    sequence_from = db.Column(db.BigInteger, nullable=False)
    sequence_to = db.Column(db.BigInteger, nullable=False)
    detected_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    resolved_at = db.Column(db.DateTime(timezone=True), nullable=True)


class TrackingEventOutbox(db.Model):
    """Outbox transactionnelle → driver.location.processed(.v3)."""

    __tablename__ = "tracking_event_outbox"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_tracking_outbox_event_id"),
        Index(
            "ix_tracking_outbox_pending",
            "driver_id",
            "published_at",
            "session_generation",
            "sequence_id",
        ),
    )

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    event_id = db.Column(db.String(64), nullable=False)
    event_type = db.Column(db.String(32), nullable=False, default="persisted")
    driver_id = db.Column(db.Integer, nullable=False, index=True)
    location_event_id = db.Column(db.String(64), nullable=False)
    session_generation = db.Column(db.BigInteger, nullable=False, default=0)
    sequence_id = db.Column(db.BigInteger, nullable=False, default=0)
    # Aligné sur la migration GPS v5 (JSONB) — évite un faux positif autogenerate JSON↔JSONB
    payload = db.Column(JSONB, nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    published_at = db.Column(db.DateTime(timezone=True), nullable=True)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    claimed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    last_error = db.Column(db.Text, nullable=True)


class DriverLocationEvent(db.Model):
    """Journal brut immuable des positions GPS (partitionné en migration SQL)."""

    __tablename__ = "driver_location_events"
    __table_args__ = (
        Index("ix_dle_driver_recorded", "driver_id", "recorded_at"),
        Index("ix_dle_driver_event", "driver_id", "location_event_id"),
    )

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    driver_id = db.Column(db.Integer, nullable=False)
    company_id = db.Column(db.Integer, nullable=False)
    location_event_id = db.Column(db.String(64), nullable=False)
    tracking_session_id = db.Column(db.String(128), nullable=False)
    session_generation = db.Column(db.BigInteger, nullable=False)
    sequence_id = db.Column(db.BigInteger, nullable=False)
    recorded_at = db.Column(db.DateTime(timezone=True), nullable=False)
    raw_latitude = db.Column(db.Float, nullable=False)
    raw_longitude = db.Column(db.Float, nullable=False)
    accuracy_m = db.Column(db.Float, nullable=True)
    speed_mps = db.Column(db.Float, nullable=True)
    heading = db.Column(db.Float, nullable=True)
    location_mode = db.Column(db.String(32), nullable=False)
    mission_id = db.Column(db.Integer, nullable=True)
    source = db.Column(db.String(32), nullable=False)
    event_payload_hash = db.Column(db.String(64), nullable=False)
    payload_schema_version = db.Column(db.String(32), nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class DriverLocationEnrichment(db.Model):
    """Enrichissement OSRM versionné — ne modifie pas le journal brut."""

    __tablename__ = "driver_location_enrichments"
    __table_args__ = (
        UniqueConstraint(
            "driver_id",
            "location_event_id",
            "enrichment_version",
            name="uq_dle_enrichment_version",
        ),
        # Ancre ledger composite (Annexe A.7) — FK soft si ledger partiel legacy
        ForeignKeyConstraint(
            ["driver_id", "location_event_id"],
            [
                "tracking_ingest_events.driver_id",
                "tracking_ingest_events.location_event_id",
            ],
            name="fk_dle_enrichment_ledger",
        ),
    )

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    driver_id = db.Column(db.Integer, nullable=False)
    location_event_id = db.Column(db.String(64), nullable=False)
    enrichment_version = db.Column(db.Integer, nullable=False, default=1)
    canonical_latitude = db.Column(db.Float, nullable=False)
    canonical_longitude = db.Column(db.Float, nullable=False)
    canonical_source = db.Column(db.String(32), nullable=False, default="osrm")
    processing_status = db.Column(db.String(16), nullable=False, default="done")
    enriched_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
