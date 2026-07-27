"""Observations shadow non autoritaires — comparateur Phase 2 (hors ledger)."""

from __future__ import annotations

from sqlalchemy import Index
from sqlalchemy.sql import func

from ext import db


class TrackingShadowObservation(db.Model):
    """Join durable direct vs shadow — ne participe ni au ledger ni au watermark."""

    __tablename__ = "tracking_shadow_observations"
    __table_args__ = (
        Index("ix_tracking_shadow_obs_expires", "expires_at"),
        Index("ix_tracking_shadow_obs_state", "comparison_state"),
        Index("ix_tracking_shadow_obs_deadline", "comparison_deadline_at"),
    )

    driver_id = db.Column(db.Integer, primary_key=True, nullable=False)
    location_event_id = db.Column(db.String(64), primary_key=True, nullable=False)
    company_id = db.Column(db.Integer, nullable=True)
    fingerprint_schema_version = db.Column(
        db.Integer, nullable=False, default=1, server_default="1"
    )

    direct_fingerprint = db.Column(db.String(128), nullable=True)
    direct_accept_status = db.Column(db.String(64), nullable=True)
    direct_accept_reason = db.Column(db.String(128), nullable=True)
    direct_seen_at = db.Column(db.DateTime(timezone=True), nullable=True)

    shadow_fingerprint = db.Column(db.String(128), nullable=True)
    shadow_accept_status = db.Column(db.String(64), nullable=True)
    shadow_accept_reason = db.Column(db.String(128), nullable=True)
    shadow_seen_at = db.Column(db.DateTime(timezone=True), nullable=True)

    comparison_deadline_at = db.Column(db.DateTime(timezone=True), nullable=True)
    comparison_state = db.Column(
        db.String(32),
        nullable=False,
        default="waiting_shadow",
        server_default="waiting_shadow",
    )
    result = db.Column(db.String(64), nullable=True)
    compared_at = db.Column(db.DateTime(timezone=True), nullable=True)
    expires_at = db.Column(db.DateTime(timezone=True), nullable=False)
    created_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
