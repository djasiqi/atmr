"""Registre durable d'un encaissement invité sans réservation complète.

Aucune donnée de carte ni jeton de paiement n'est stocké.
Un enregistrement n'est pas un remboursement.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from ext import db

STATE_UNCERTAIN = "uncertain"
STATE_CAPTURED = "captured"
STATE_FAILED = "failed"
STATE_RESOLVED = "resolved"

FINANCIAL_STATES = (
    STATE_UNCERTAIN,
    STATE_CAPTURED,
    STATE_FAILED,
    STATE_RESOLVED,
)


class GuestPaymentIncident(db.Model):
    """Une opération de paiement invité, retrouvable après expiration du dossier temporaire."""

    __tablename__ = "guest_payment_incident"
    __table_args__ = (
        CheckConstraint(
            "financial_state IN ('uncertain', 'captured', 'failed', 'resolved')",
            name="ck_guest_payment_incident_state",
        ),
        CheckConstraint(
            "currency = 'CHF'",
            name="ck_guest_payment_incident_currency",
        ),
        CheckConstraint(
            "amount >= 0",
            name="ck_guest_payment_incident_amount",
        ),
        Index("ix_guest_payment_incident_state", "financial_state"),
        Index("ix_guest_payment_incident_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        default=lambda: str(uuid.uuid4()),
    )
    guest_booking_id: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    saferpay_transaction_id: Mapped[str | None] = mapped_column(
        String(128), nullable=True, unique=True
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(
        String(3), nullable=False, server_default=text("'CHF'")
    )
    financial_state: Mapped[str] = mapped_column(String(16), nullable=False)
    booking_ids_text: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'[]'")
    )
    recovery_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
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
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    events: Mapped[list[GuestPaymentIncidentEvent]] = relationship(
        "GuestPaymentIncidentEvent",
        back_populates="incident",
        order_by="GuestPaymentIncidentEvent.id",
    )


class GuestPaymentIncidentEvent(db.Model):
    """Piste d'audit append-only. Le client et le chauffeur ne peuvent pas l'écrire."""

    __tablename__ = "guest_payment_incident_event"
    __table_args__ = (Index("ix_guest_payment_incident_event_incident", "incident_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    incident_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("guest_payment_incident.id", ondelete="RESTRICT"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=True
    )
    payload_text: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'{}'")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    incident: Mapped[GuestPaymentIncident] = relationship(
        "GuestPaymentIncident", back_populates="events"
    )
