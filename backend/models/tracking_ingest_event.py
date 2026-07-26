"""Ledger d'idempotence GPS F-02 — autorité PostgreSQL."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Index, UniqueConstraint
from sqlalchemy.sql import func

from ext import db


class TrackingIngestEvent(db.Model):
    """Événement GPS durablement accepté (exactly-once logique)."""

    __tablename__ = "tracking_ingest_events"
    __table_args__ = (
        UniqueConstraint(
            "driver_id",
            "location_event_id",
            name="uq_tracking_ingest_driver_event",
        ),
        Index("ix_tracking_ingest_received_at", "received_at"),
    )

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    driver_id = db.Column(
        db.Integer, db.ForeignKey("driver.id"), nullable=False, index=True
    )
    company_id = db.Column(
        db.Integer, db.ForeignKey("company.id"), nullable=False, index=True
    )
    location_event_id = db.Column(db.String(64), nullable=False)
    event_payload_hash = db.Column(db.String(64), nullable=False)
    payload_schema_version = db.Column(
        db.String(32), nullable=False, default="tracking-event-payload-v1"
    )
    source = db.Column(db.String(32), nullable=False)
    recorded_at = db.Column(db.DateTime(timezone=True), nullable=False)
    received_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class TrackingDerivedRepairPending(db.Model):
    """File de réparation Redis canonical (écrite dans la TX principale)."""

    __tablename__ = "tracking_derived_repair_pending"
    __table_args__ = (
        UniqueConstraint(
            "driver_id",
            "location_event_id",
            "repair_kind",
            name="uq_tracking_derived_repair",
        ),
    )

    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    driver_id = db.Column(db.Integer, nullable=False, index=True)
    location_event_id = db.Column(db.String(64), nullable=False)
    repair_kind = db.Column(
        db.String(32), nullable=False, default="redis_canonical"
    )
    target_recorded_at = db.Column(db.DateTime(timezone=True), nullable=False)
    target_sequence_id = db.Column(db.BigInteger, nullable=True)
    status = db.Column(db.String(16), nullable=False, default="pending")
    attempts = db.Column(db.Integer, nullable=False, default=0)
    last_error = db.Column(db.Text, nullable=True)
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
