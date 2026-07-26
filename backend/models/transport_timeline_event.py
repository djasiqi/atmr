"""Journal immutable des événements transport institution (timeline unifiée)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    SmallInteger,
    String,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class TransportTimelineEvent(db.Model):
    __tablename__ = "transport_timeline_events"
    __table_args__ = (
        Index("ix_tte_request_created", "transport_request_id", "created_at"),
        Index("ix_tte_booking_created", "booking_id", "created_at"),
        Index("ix_tte_institution_created", "institution_id", "created_at"),
        Index("ix_tte_source_event", "source_event_id"),
        Index("ix_tte_correlation", "correlation_id"),
        Index("ix_tte_event_type", "event_type"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    transport_request_id: Mapped[int | None] = mapped_column(
        ForeignKey("transport_requests.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    booking_id: Mapped[int | None] = mapped_column(
        ForeignKey("booking.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    institution_id: Mapped[int | None] = mapped_column(
        ForeignKey("institutions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    company_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="SET NULL"),
        nullable=True,
    )
    driver_id: Mapped[int | None] = mapped_column(
        ForeignKey("driver.id", ondelete="SET NULL"),
        nullable=True,
    )

    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    payload_version: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default="1"
    )
    correlation_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    source_event_id: Mapped[int | None] = mapped_column(
        ForeignKey("transport_timeline_events.id", ondelete="SET NULL"),
        nullable=True,
    )

    created_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    source_event = relationship(
        "TransportTimelineEvent",
        remote_side=[id],
        foreign_keys=[source_event_id],
    )

    def serialize(self) -> dict[str, Any]:
        from services.institutions.transport_timeline_service import (
            build_timeline_label,
        )

        return {
            "id": self.id,
            "transport_request_id": self.transport_request_id,
            "booking_id": self.booking_id,
            "institution_id": self.institution_id,
            "event_type": self.event_type,
            "label": build_timeline_label(self),
            "actor_type": self.actor_type,
            "actor_user_id": self.actor_user_id,
            "company_id": self.company_id,
            "driver_id": self.driver_id,
            "payload": self.payload,
            "payload_version": self.payload_version,
            "correlation_id": self.correlation_id,
            "source_event_id": self.source_event_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
