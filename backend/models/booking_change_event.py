"""Événements métier de modification de réservation (audit trail institution/ops)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class BookingChangeEvent(db.Model):
    __tablename__ = "booking_change_events"
    __table_args__ = (
        Index("ix_bce_booking_created", "booking_id", "created_at"),
        Index("ix_bce_severity_ack_created", "severity", "ack_required", "created_at"),
        Index("ix_bce_correlation", "correlation_id"),
        Index("ix_bce_institution_created", "institution_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    booking_id: Mapped[int] = mapped_column(
        ForeignKey("booking.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    transport_request_id: Mapped[int | None] = mapped_column(
        ForeignKey("transport_requests.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    institution_id: Mapped[int | None] = mapped_column(
        ForeignKey("institutions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    booking_version: Mapped[int] = mapped_column(Integer, nullable=False)

    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    actor_role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_display_name: Mapped[str | None] = mapped_column(String(200), nullable=True)

    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    change_class: Mapped[str] = mapped_column(String(16), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)

    before_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    after_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    changed_fields: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    change_scope: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    operational_impact: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )

    financial_actor_role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    billing_change_reason_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )

    ack_required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )

    correlation_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    created_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    acknowledgements = relationship(
        "BookingChangeAcknowledgement",
        back_populates="event",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def serialize(self) -> dict[str, Any]:
        acks = list(self.acknowledgements or [])
        return {
            "id": self.id,
            "booking_id": self.booking_id,
            "transport_request_id": self.transport_request_id,
            "institution_id": self.institution_id,
            "booking_version": self.booking_version,
            "actor_user_id": self.actor_user_id,
            "actor_role": self.actor_role,
            "actor_type": self.actor_type,
            "actor_display_name": self.actor_display_name,
            "action_type": self.action_type,
            "change_class": self.change_class,
            "severity": self.severity,
            "before_snapshot": self.before_snapshot,
            "after_snapshot": self.after_snapshot,
            "changed_fields": self.changed_fields,
            "reason": self.reason,
            "change_scope": self.change_scope,
            "source": self.source,
            "operational_impact": self.operational_impact,
            "financial_actor_role": self.financial_actor_role,
            "billing_change_reason_code": self.billing_change_reason_code,
            "ack_required": self.ack_required,
            "ack_received_count": len(acks),
            "acknowledgements": [a.serialize() for a in acks],
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BookingChangeAcknowledgement(db.Model):
    __tablename__ = "booking_change_acknowledgements"
    __table_args__ = (
        Index(
            "uq_bce_ack_event_user_actor",
            "event_id",
            "user_id",
            "actor_type",
            unique=True,
        ),
        Index("ix_bce_ack_event", "event_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    event_id: Mapped[int] = mapped_column(
        ForeignKey("booking_change_events.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    ack_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    ack_channel: Mapped[str | None] = mapped_column(String(32), nullable=True)
    ack_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    event = relationship("BookingChangeEvent", back_populates="acknowledgements")

    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "user_id": self.user_id,
            "actor_type": self.actor_type,
            "ack_at": self.ack_at.isoformat() if self.ack_at else None,
            "ack_channel": self.ack_channel,
            "ack_metadata": self.ack_metadata,
        }
