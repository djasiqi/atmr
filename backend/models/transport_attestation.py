"""Attestation factuelle d'un trajet réalisé.

Une ligne décrit un seul trajet. Elle ne crée ni facture, ni encaissement,
ni reconnaissance de dette. La signature, lorsqu'elle existe, est celle
demandée au passager transporté.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ext import db

OUTCOME_SIGNATURE = "signature_obtained"
OUTCOME_REFUSED = "signature_refused"
OUTCOME_IMPOSSIBLE = "signature_impossible"
OUTCOME_REVIEW = "pending_verification"

ATTESTATION_OUTCOMES = (
    OUTCOME_SIGNATURE,
    OUTCOME_REFUSED,
    OUTCOME_IMPOSSIBLE,
    OUTCOME_REVIEW,
)

REASON_REFUSED = "refused"
REASON_UNABLE = "unable"
REASON_TECHNICAL = "technical"

ATTESTATION_REASONS = (REASON_REFUSED, REASON_UNABLE, REASON_TECHNICAL)


class TransportAttestation(db.Model):
    """Preuve de réalisation d'un trajet, confirmée une fois."""

    __tablename__ = "transport_attestation"
    __table_args__ = (
        CheckConstraint(
            "outcome IN ("
            "'signature_obtained', 'signature_refused', "
            "'signature_impossible', 'pending_verification')",
            name="ck_transport_attestation_outcome",
        ),
        CheckConstraint(
            "reason_code IS NULL OR reason_code IN ('refused', 'unable', 'technical')",
            name="ck_transport_attestation_reason",
        ),
        CheckConstraint(
            "(outcome = 'signature_obtained' AND signature_sha256 IS NOT NULL) "
            "OR (outcome <> 'signature_obtained' AND signature_sha256 IS NULL)",
            name="ck_transport_attestation_signature",
        ),
        Index("ix_transport_attestation_company", "owner_company_id"),
        Index("ix_transport_attestation_driver", "driver_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        default=lambda: str(uuid.uuid4()),
    )
    booking_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
        index=True,
    )
    owner_company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="RESTRICT"), nullable=False
    )
    executing_company_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("company.id", ondelete="RESTRICT"), nullable=True
    )
    driver_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("driver.id", ondelete="RESTRICT"), nullable=False
    )
    passenger_name: Mapped[str] = mapped_column(String(120), nullable=False)
    pickup_location: Mapped[str] = mapped_column(String(500), nullable=False)
    dropoff_location: Mapped[str] = mapped_column(String(500), nullable=False)
    scheduled_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    arrived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    boarded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    reason_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    signature_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    signature_b64: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(80), nullable=False)
    confirmed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    events: Mapped[list[TransportAttestationEvent]] = relationship(
        back_populates="attestation",
        order_by="TransportAttestationEvent.id",
    )


class TransportAttestationEvent(db.Model):
    """Trace append-only. Ne contient pas l'image de signature."""

    __tablename__ = "transport_attestation_event"
    __table_args__ = (
        Index("ix_transport_attestation_event_parent", "attestation_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    attestation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("transport_attestation.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    driver_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detail: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'{}'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    attestation: Mapped[TransportAttestation] = relationship(back_populates="events")
