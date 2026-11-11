"""Registre durable d'un imprévu opérationnel.

Distinct du registre des incidents de paiement. Un signalement ne libère
pas l'entreprise, ne modifie pas un paiement et ne déclenche pas de
remboursement.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
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

STATE_SIGNALED = "signale"
STATE_ANALYSIS = "analyse"
STATE_RESOLVED = "resolu"
STATE_CLOSED = "cloture"
OPEN_STATES = (STATE_SIGNALED, STATE_ANALYSIS)


class OperationalIncident(db.Model):
    """Signalement conservé après redémarrage."""

    __tablename__ = "operational_incident"
    __table_args__ = (
        CheckConstraint(
            "state IN ('signale', 'analyse', 'resolu', 'cloture')",
            name="ck_operational_incident_state",
        ),
        CheckConstraint(
            "state <> 'cloture' OR ("
            "close_justification IS NOT NULL "
            "AND length(btrim(close_justification)) > 0)",
            name="ck_operational_incident_close_reason",
        ),
        CheckConstraint(
            "formally_released = false",
            name="ck_operational_incident_not_released",
        ),
        Index("ix_operational_incident_state", "state"),
        Index("ix_operational_incident_booking", "booking_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        default=lambda: str(uuid.uuid4()),
    )
    idempotency_key: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True
    )
    booking_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("booking.id", ondelete="RESTRICT"),
        nullable=False,
    )
    owner_company_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    actor_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=True
    )
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False)
    motif: Mapped[str] = mapped_column(Text, nullable=False)
    group_ids_text: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'[]'")
    )
    completed_ids_text: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'[]'")
    )
    remaining_ids_text: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'[]'")
    )
    state: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text("'signale'")
    )
    close_justification: Mapped[str | None] = mapped_column(Text, nullable=True)
    formally_released: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    events: Mapped[list[OperationalIncidentEvent]] = relationship(
        "OperationalIncidentEvent",
        back_populates="incident",
        order_by="OperationalIncidentEvent.id",
    )


class OperationalIncidentEvent(db.Model):
    """Piste d'audit du signalement. Elle n'autorise aucune libération."""

    __tablename__ = "operational_incident_event"
    __table_args__ = (Index("ix_operational_incident_event_incident", "incident_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    incident_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("operational_incident.id", ondelete="RESTRICT"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    detail: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    incident: Mapped[OperationalIncident] = relationship(
        "OperationalIncident", back_populates="events"
    )
