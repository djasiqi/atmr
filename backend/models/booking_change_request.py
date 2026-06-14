"""Demandes de modification critique nécessitant confirmation transporteur."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
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


class BookingChangeRequestStatus:
    PENDING = "pending"
    ACCEPTED = "accepted"
    REFUSED = "refused"
    EXPIRED = "expired"
    ESCALATION_REQUIRED = "escalation_required"
    SUPERSEDED = "superseded"

    ALL = frozenset(
        {
            PENDING,
            ACCEPTED,
            REFUSED,
            EXPIRED,
            ESCALATION_REQUIRED,
            SUPERSEDED,
        }
    )


class BookingChangeRequest(db.Model):
    __tablename__ = "booking_change_requests"
    __table_args__ = (
        Index("ix_bcr_booking_status", "booking_id", "status"),
        Index("ix_bcr_institution_created", "institution_id", "created_at"),
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
    )
    institution_id: Mapped[int] = mapped_column(
        ForeignKey("institutions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")

    proposed_patch: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    before_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    after_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    changed_fields: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    requested_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    requested_by_role: Mapped[str | None] = mapped_column(String(64), nullable=True)

    responded_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
    )
    responded_by_role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    responded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    booking = relationship(
        "Booking",
        back_populates="change_requests",
        foreign_keys=[booking_id],
    )

    def serialize(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "booking_id": self.booking_id,
            "transport_request_id": self.transport_request_id,
            "institution_id": self.institution_id,
            "status": self.status,
            "version": self.version,
            "proposed_patch": self.proposed_patch,
            "before_snapshot": self.before_snapshot,
            "after_snapshot": self.after_snapshot,
            "changed_fields": self.changed_fields,
            "reason": self.reason,
            "requested_by_user_id": self.requested_by_user_id,
            "requested_by_role": self.requested_by_role,
            "responded_by_user_id": self.responded_by_user_id,
            "responded_by_role": self.responded_by_role,
            "responded_at": self.responded_at.isoformat() if self.responded_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
