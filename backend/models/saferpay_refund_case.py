"""Dossier de remboursement Saferpay.

Une demande créée n'est pas approuvée.
Une demande approuvée n'est pas une opération envoyée.
Une opération envoyée n'est pas une opération confirmée.
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

REQUEST_CREATED = "created"
REQUEST_APPROVED = "approved"
REQUEST_REFUSED = "refused"

EXEC_NOT_SENT = "not_sent"
EXEC_SENT = "sent"
EXEC_CONFIRMED = "confirmed"
EXEC_FAILED = "failed"
EXEC_UNCERTAIN = "uncertain"


class SaferpayRefundCase(db.Model):
    """Dossier lié à un paiement Saferpay capturé. Aucun appel prestataire ici."""

    __tablename__ = "saferpay_refund_case"
    __table_args__ = (
        CheckConstraint(
            "request_status IN ('created', 'approved', 'refused')",
            name="ck_saferpay_refund_request_status",
        ),
        CheckConstraint(
            "execution_status IN ('not_sent', 'sent', 'confirmed', 'failed', 'uncertain')",
            name="ck_saferpay_refund_execution_status",
        ),
        CheckConstraint("currency = 'CHF'", name="ck_saferpay_refund_currency_chf"),
        CheckConstraint(
            "captured_amount > 0", name="ck_saferpay_refund_captured_positive"
        ),
        CheckConstraint(
            "proposed_amount > 0", name="ck_saferpay_refund_proposed_positive"
        ),
        CheckConstraint(
            "proposed_amount <= captured_amount",
            name="ck_saferpay_refund_proposed_within_capture",
        ),
        CheckConstraint(
            "remaining_at_open >= 0",
            name="ck_saferpay_refund_remaining_nonneg",
        ),
        CheckConstraint(
            "prior_confirmed_amount >= 0",
            name="ck_saferpay_refund_prior_nonneg",
        ),
        CheckConstraint(
            "proposed_amount <= remaining_at_open",
            name="ck_saferpay_refund_proposed_within_remaining",
        ),
        CheckConstraint(
            "confirmed_amount IS NULL OR "
            "(confirmed_amount > 0 AND confirmed_amount <= proposed_amount)",
            name="ck_saferpay_refund_confirmed_within_proposed",
        ),
        CheckConstraint(
            "("
            "request_status = 'created' AND approver_user_id IS NULL"
            ") OR ("
            "request_status IN ('approved', 'refused') AND approver_user_id IS NOT NULL"
            ")",
            name="ck_saferpay_refund_decision_approver",
        ),
        CheckConstraint(
            "execution_status <> 'confirmed' OR ("
            "provider_refund_reference IS NOT NULL AND confirmed_amount IS NOT NULL"
            ")",
            name="ck_saferpay_refund_confirmed_has_proof",
        ),
        Index("ix_saferpay_refund_case_payment", "payment_id"),
        Index("ix_saferpay_refund_case_booking", "booking_id"),
        Index(
            "uq_saferpay_refund_provider_reference",
            "provider_refund_reference",
            unique=True,
            postgresql_where=text("provider_refund_reference IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        unique=True,
        default=lambda: str(uuid.uuid4()),
    )
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="RESTRICT"), nullable=False
    )
    payment_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("payment.id", ondelete="RESTRICT"), nullable=False
    )
    holder_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    holder_public_id: Mapped[str] = mapped_column(String(64), nullable=False)
    holder_label: Mapped[str] = mapped_column(String(255), nullable=False)
    saferpay_transaction_id: Mapped[str] = mapped_column(String(128), nullable=False)
    captured_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CHF")
    prior_confirmed_amount: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=Decimal("0.00")
    )
    remaining_at_open: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    proposed_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    eligibility_justification: Mapped[str] = mapped_column(Text, nullable=False)
    request_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=REQUEST_CREATED,
        server_default=REQUEST_CREATED,
    )
    approver_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=True
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=EXEC_NOT_SENT, server_default=EXEC_NOT_SENT
    )
    operation_reference: Mapped[str | None] = mapped_column(
        String(64), nullable=True, unique=True
    )
    provider_refund_reference: Mapped[str | None] = mapped_column(
        String(128), nullable=True
    )
    confirmed_amount: Mapped[Decimal | None] = mapped_column(
        Numeric(12, 2), nullable=True
    )
    confirmed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    proof_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    events: Mapped[list[SaferpayRefundCaseEvent]] = relationship(
        "SaferpayRefundCaseEvent",
        back_populates="case",
        order_by="SaferpayRefundCaseEvent.id",
    )


class SaferpayRefundCaseEvent(db.Model):
    """Journal d'audit append-only. Le client ne peut pas le réécrire."""

    __tablename__ = "saferpay_refund_case_event"
    __table_args__ = (Index("ix_saferpay_refund_event_case", "case_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("saferpay_refund_case.id", ondelete="RESTRICT"),
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

    case: Mapped[SaferpayRefundCase] = relationship(
        "SaferpayRefundCase", back_populates="events"
    )
