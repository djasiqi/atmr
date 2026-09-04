"""Contestation de prestation — historique, preuves, résolution (jamais un DELETE)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class BookingDispute(db.Model):
    __tablename__ = "booking_disputes"
    __table_args__ = (
        Index("ix_booking_disputes_booking_status", "booking_id", "status"),
        Index("ix_booking_disputes_company_status", "company_id", "status"),
        Index("ix_booking_disputes_institution_status", "institution_id", "status"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    booking_id: Mapped[int] = mapped_column(
        ForeignKey("booking.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    company_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    institution_id: Mapped[int | None] = mapped_column(
        ForeignKey("institutions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    opened_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    opened_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    institution_reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    institution_reason_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    frozen_amount_ht: Mapped[Any] = mapped_column(Numeric(12, 2), nullable=True)
    frozen_payer_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    frozen_billing_party_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    carrier_stance: Mapped[str | None] = mapped_column(String(40), nullable=True)
    carrier_exclusion_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    carrier_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    carrier_responded_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    carrier_responded_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )

    proposed_amount_ht: Mapped[Any] = mapped_column(Numeric(12, 2), nullable=True)
    proposed_payer_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    proposed_correction_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    submitted_at: Mapped[Any] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[Any] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    resolver_role: Mapped[str | None] = mapped_column(String(40), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    booking = relationship("Booking", foreign_keys=[booking_id])
    evidence = relationship(
        "BookingDisputeEvidence",
        back_populates="dispute",
        cascade="all, delete-orphan",
        order_by="BookingDisputeEvidence.id",
    )
    events = relationship(
        "BookingDisputeEvent",
        back_populates="dispute",
        cascade="all, delete-orphan",
        order_by="BookingDisputeEvent.id",
    )


class BookingDisputeEvidence(db.Model):
    __tablename__ = "booking_dispute_evidence"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    dispute_id: Mapped[int] = mapped_column(
        ForeignKey("booking_disputes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    stored_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    original_filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )

    dispute = relationship("BookingDispute", back_populates="evidence")


class BookingDisputeEvent(db.Model):
    __tablename__ = "booking_dispute_events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    dispute_id: Mapped[int] = mapped_column(
        ForeignKey("booking_disputes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    actor_role: Mapped[str | None] = mapped_column(String(40), nullable=True)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    dispute = relationship("BookingDispute", back_populates="events")
