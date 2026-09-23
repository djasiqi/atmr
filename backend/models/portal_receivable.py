"""Créance PORTAL : facture réelle du transporteur, hors estimation de course.

Ce n'est pas le moteur ``Invoice`` entreprise (S1/S2). Le montant vient de la
facture externe du transporteur, jamais de ``booking.amount``. Aucun
``PAYMENT_HOLD`` n'est déclenché depuis cette table.
"""

from __future__ import annotations

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
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db

RECEIVABLE_ISSUED = "issued"
RECEIVABLE_PARTIALLY_PAID = "partially_paid"
RECEIVABLE_PAID = "paid"
RECEIVABLE_OVERDUE = "overdue"
RECEIVABLE_DISPUTED = "disputed"
RECEIVABLE_CANCELLED = "cancelled"

RECEIVABLE_STATUSES = (
    RECEIVABLE_ISSUED,
    RECEIVABLE_PARTIALLY_PAID,
    RECEIVABLE_PAID,
    RECEIVABLE_OVERDUE,
    RECEIVABLE_DISPUTED,
    RECEIVABLE_CANCELLED,
)

PAYMENT_BANK_TRANSFER = "bank_transfer"
PAYMENT_CASH = "cash"
PAYMENT_CARD = "card"
PAYMENT_OTHER = "other"

PAYMENT_METHODS = (
    PAYMENT_BANK_TRANSFER,
    PAYMENT_CASH,
    PAYMENT_CARD,
    PAYMENT_OTHER,
)


class PortalReceivable(db.Model):
    """Créance d'un transporteur envers un client privé."""

    __tablename__ = "portal_receivable"
    __table_args__ = (
        UniqueConstraint(
            "creditor_company_id",
            "external_invoice_number",
            name="uq_portal_receivable_creditor_invoice",
        ),
        CheckConstraint(
            "status IN ("
            "'issued', 'partially_paid', 'paid', 'overdue', 'disputed', 'cancelled'"
            ")",
            name="ck_portal_receivable_status",
        ),
        CheckConstraint("total_amount >= 0", name="ck_portal_receivable_total_nonneg"),
        CheckConstraint("amount_paid >= 0", name="ck_portal_receivable_paid_nonneg"),
        CheckConstraint(
            "balance_due >= 0", name="ck_portal_receivable_balance_nonneg"
        ),
        Index("ix_portal_receivable_debtor", "debtor_user_id"),
        Index("ix_portal_receivable_due_date", "due_date"),
        Index("ix_portal_receivable_status", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    creditor_company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    creditor_name_snapshot: Mapped[str] = mapped_column(String(200), nullable=False)

    debtor_user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("user.id", ondelete="RESTRICT"),
        nullable=False,
    )
    debtor_name_snapshot: Mapped[str] = mapped_column(String(200), nullable=False)
    debtor_email_snapshot: Mapped[str | None] = mapped_column(String(255), nullable=True)
    debtor_phone_snapshot: Mapped[str | None] = mapped_column(String(40), nullable=True)
    debtor_billing_address_snapshot: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )

    external_invoice_number: Mapped[str] = mapped_column(String(80), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CHF")
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    due_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    total_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    amount_paid: Mapped[Decimal] = mapped_column(
        Numeric(12, 2), nullable=False, default=Decimal("0.00")
    )
    balance_due: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)

    status: Mapped[str] = mapped_column(String(32), nullable=False, default=RECEIVABLE_ISSUED)

    disputed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    dispute_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    disputed_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )

    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancelled_by_user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    cancellation_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    recorded_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
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

    lines = relationship(
        "PortalReceivableLine",
        back_populates="receivable",
        cascade="all, delete-orphan",
        order_by="PortalReceivableLine.id",
    )
    payments = relationship(
        "PortalReceivablePayment",
        back_populates="receivable",
        cascade="all, delete-orphan",
        order_by="PortalReceivablePayment.id",
    )


class PortalReceivableLine(db.Model):
    """Ligne de créance liée à une course et à son BOOKING_CREATED."""

    __tablename__ = "portal_receivable_line"
    __table_args__ = (
        CheckConstraint(
            "invoiced_amount >= 0", name="ck_portal_receivable_line_amount_nonneg"
        ),
        Index("ix_portal_receivable_line_booking", "booking_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    receivable_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("portal_receivable.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    booking_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("booking.id", ondelete="RESTRICT"), nullable=False
    )
    booking_contract_event_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("client_booking_contract_event.id", ondelete="RESTRICT"),
        nullable=False,
    )
    invoiced_amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500), nullable=True)

    receivable = relationship("PortalReceivable", back_populates="lines")


class PortalReceivablePayment(db.Model):
    """Paiement hors plateforme enregistré par le transporteur."""

    __tablename__ = "portal_receivable_payment"
    __table_args__ = (
        CheckConstraint(
            "amount > 0", name="ck_portal_receivable_payment_amount_positive"
        ),
        CheckConstraint(
            "method IN ('bank_transfer', 'cash', 'card', 'other')",
            name="ck_portal_receivable_payment_method",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    receivable_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("portal_receivable.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    amount: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
    paid_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    method: Mapped[str] = mapped_column(String(32), nullable=False)
    reference: Mapped[str | None] = mapped_column(String(120), nullable=True)
    recorded_by_user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("user.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    receivable = relationship("PortalReceivable", back_populates="payments")
