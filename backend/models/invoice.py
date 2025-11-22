# models/invoice.py
"""Models Invoice et tous ses modèles liés (lignes, paiements, rappels, etc.).
Extrait depuis models.py (lignes ~1763-3258).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
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
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing_extensions import override

from ext import db

from .base import _as_bool, _iso
from .enums import InvoiceLineType, InvoiceStatus, PaymentMethod


class Invoice(db.Model):
    """Modèle principal pour les factures."""

    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id"), nullable=False, index=True
    )
    client_id: Mapped[int] = mapped_column(
        ForeignKey("client.id"), nullable=False, index=True
    )

    # Facturation tierce (Third-Party Billing)
    bill_to_client_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("client.id"), nullable=True, index=True
    )

    # Période de facturation
    period_month: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-12
    period_year: Mapped[int] = mapped_column(Integer, nullable=False)

    # Numéro de facture unique par entreprise
    invoice_number: Mapped[str] = mapped_column(String(50), nullable=False)
    currency = Column(String(3), default="CHF", nullable=False)

    # Montants
    subtotal_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    late_fee_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    reminder_fee_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    vat_total_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    vat_breakdown = Column(JSONB, nullable=True)
    total_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    amount_paid: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    balance_due: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )

    # Dates clés
    issued_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    due_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    paid_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # Statut
    status = Column(
        SAEnum(InvoiceStatus, name="invoice_status"),
        nullable=False,
        default=InvoiceStatus.DRAFT,
    )

    # Rappels
    reminder_level = Column(Integer, nullable=False, default=0)  # 0 = aucun, 1, 2, 3
    last_reminder_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Artifacts
    pdf_url: Mapped[str] = mapped_column(String(500), nullable=True)
    qr_reference: Mapped[str] = mapped_column(String(50), nullable=True)
    meta = Column(JSONB, nullable=True)

    # Relations
    company = relationship("Company", backref="invoices")
    client = relationship(
        "Client", foreign_keys=[client_id], backref="service_invoices"
    )
    bill_to_client = relationship(
        "Client", foreign_keys=[bill_to_client_id], backref="billing_invoices"
    )
    lines = relationship(
        "InvoiceLine", back_populates="invoice", cascade="all, delete-orphan"
    )
    payments = relationship(
        "InvoicePayment", back_populates="invoice", cascade="all, delete-orphan"
    )
    reminders = relationship(
        "InvoiceReminder", back_populates="invoice", cascade="all, delete-orphan"
    )

    # Index et contraintes
    __table_args__ = (
        UniqueConstraint(
            "company_id", "invoice_number", name="uq_company_invoice_number"
        ),
        Index("ix_invoice_company_period", "company_id", "period_year", "period_month"),
        Index("ix_invoice_status", "company_id", "status"),
        Index("ix_invoice_due_date", "due_date"),
        CheckConstraint("total_amount >= 0", name="chk_invoice_amount_positive"),
        CheckConstraint("balance_due >= 0", name="chk_invoice_balance_nonneg"),
        CheckConstraint("amount_paid >= 0", name="chk_invoice_paid_nonneg"),
    )

    @override
    def __repr__(self):
        return f"<Invoice {self.invoice_number} - {self.status.value}>"

    @property
    def is_overdue(self):
        """Vérifie si la facture est en retard."""
        return (
            self.balance_due > 0
            and self.due_date is not None
            and datetime.now(UTC) > self.due_date
        )

    def update_balance(self):
        """Met à jour le solde et le statut basé sur les paiements."""
        self.amount_paid = sum(payment.amount for payment in self.payments)
        self.balance_due = self.total_amount - self.amount_paid

        # Mise à jour du statut
        if self.balance_due <= 0:
            self.status = InvoiceStatus.PAID
            self.paid_at = datetime.now(UTC)
        elif self.amount_paid > 0:
            self.status = InvoiceStatus.PARTIALLY_PAID
        elif self.is_overdue:
            self.status = InvoiceStatus.OVERDUE

    def mark_as_paid(self):
        """Marque la facture comme payée."""
        self.status = InvoiceStatus.PAID
        self.paid_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def mark_as_sent(self):
        """Marque la facture comme envoyée."""
        self.status = InvoiceStatus.SENT
        self.sent_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def cancel(self):
        """Annule la facture."""
        self.status = InvoiceStatus.CANCELLED
        self.cancelled_at = datetime.now(UTC)
        self.updated_at = datetime.now(UTC)

    def to_dict(self):
        """Sérialise la facture en dictionnaire."""
        return {
            "id": self.id,
            "company_id": self.company_id,
            "client_id": self.client_id,
            "bill_to_client_id": self.bill_to_client_id,
            "period_month": self.period_month,
            "period_year": self.period_year,
            "invoice_number": self.invoice_number,
            "currency": self.currency,
            "subtotal_amount": float(self.subtotal_amount),
            "late_fee_amount": float(self.late_fee_amount),
            "reminder_fee_amount": float(self.reminder_fee_amount),
            "vat_total_amount": float(self.vat_total_amount),
            "vat_breakdown": self.vat_breakdown,
            "total_amount": float(self.total_amount),
            "amount_paid": float(self.amount_paid),
            "balance_due": float(self.balance_due),
            "issued_at": _iso(self.issued_at),
            "due_date": _iso(self.due_date),
            "sent_at": _iso(self.sent_at),
            "paid_at": _iso(self.paid_at),
            "cancelled_at": _iso(self.cancelled_at),
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "status": self.status.value,
            "reminder_level": self.reminder_level,
            "last_reminder_at": _iso(self.last_reminder_at),
            "pdf_url": self.pdf_url,
            "qr_reference": self.qr_reference,
            "meta": self.meta,
            "client": {
                "id": self.client.id,
                "first_name": getattr(self.client.user, "first_name", "")
                if hasattr(self.client, "user") and self.client.user
                else "",
                "last_name": getattr(self.client.user, "last_name", "")
                if hasattr(self.client, "user") and self.client.user
                else "",
                "username": getattr(self.client.user, "username", "")
                if hasattr(self.client, "user") and self.client.user
                else "",
                "is_institution": _as_bool(self.client.is_institution)
                if self.client
                else False,
                "institution_name": self.client.institution_name
                if self.client
                else None,
            }
            if self.client
            else None,
            "bill_to_client": {
                "id": self.bill_to_client.id,
                "first_name": getattr(self.bill_to_client.user, "first_name", "")
                if hasattr(self.bill_to_client, "user") and self.bill_to_client.user
                else "",
                "last_name": getattr(self.bill_to_client.user, "last_name", "")
                if hasattr(self.bill_to_client, "user") and self.bill_to_client.user
                else "",
                "username": getattr(self.bill_to_client.user, "username", "")
                if hasattr(self.bill_to_client, "user") and self.bill_to_client.user
                else "",
                "is_institution": _as_bool(self.bill_to_client.is_institution),
                "institution_name": self.bill_to_client.institution_name,
                "billing_address": self.bill_to_client.billing_address,
                "contact_email": self.bill_to_client.contact_email,
            }
            if self.bill_to_client
            else None,
            "lines": [line.to_dict() for line in self.lines]
            if hasattr(self, "lines")
            else [],
            "payments": [payment.to_dict() for payment in self.payments]
            if hasattr(self, "payments")
            else [],
            "reminders": [reminder.to_dict() for reminder in self.reminders]
            if hasattr(self, "reminders")
            else [],
        }


class InvoiceLine(db.Model):
    """Lignes de facture."""

    __tablename__ = "invoice_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), nullable=False)

    type: Mapped[InvoiceLineType] = mapped_column(
        SAEnum(InvoiceLineType, name="invoice_line_type"), nullable=False
    )
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    qty: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False, default=1)
    unit_price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    line_total: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    vat_rate: Mapped[Optional[Decimal]] = mapped_column(Numeric(5, 2), nullable=True)
    vat_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    total_with_vat: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    adjustment_note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Optionnel : tracer la source (réservation)
    reservation_id = Column(
        Integer,
        ForeignKey("booking.id", name="fk_invoice_line_reservation"),
        nullable=True,
    )

    # Relations
    invoice = relationship("Invoice", back_populates="lines")
    reservation = relationship(
        "Booking",
        foreign_keys=[reservation_id],
        backref="invoice_lines_for_reservation",
    )

    @override
    def __repr__(self):
        return f"<InvoiceLine {self.description} - {self.line_total} CHF>"

    def to_dict(self):
        """Sérialise la ligne de facture en dictionnaire."""
        return {
            "id": self.id,
            "invoice_id": self.invoice_id,
            "type": self.type.value,
            "description": self.description,
            "qty": float(self.qty),
            "unit_price": float(self.unit_price),
            "line_total": float(self.line_total),
            "vat_rate": float(self.vat_rate) if self.vat_rate is not None else None,
            "vat_amount": float(self.vat_amount),
            "total_with_vat": float(self.total_with_vat),
            "adjustment_note": self.adjustment_note,
            "reservation_id": self.reservation_id,
        }


class InvoicePayment(db.Model):
    """Paiements des factures."""

    __tablename__ = "invoice_payments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), nullable=False)

    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    paid_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    method = Column(
        SAEnum(
            PaymentMethod,
            name="payment_method",
            values_callable=lambda enum_cls: [e.value for e in enum_cls],
        ),
        nullable=False,
    )
    reference: Mapped[str] = mapped_column(String(100), nullable=True)

    # Relations
    invoice = relationship("Invoice", back_populates="payments")

    @override
    def __repr__(self):
        return f"<InvoicePayment {self.amount} CHF - {self.method.value}>"

    def to_dict(self):
        """Sérialise le paiement en dictionnaire."""
        return {
            "id": self.id,
            "invoice_id": self.invoice_id,
            "amount": float(self.amount),
            "paid_at": _iso(self.paid_at),
            "method": self.method.value,
            "reference": self.reference,
        }


class InvoiceReminder(db.Model):
    """Rappels de facture."""

    __tablename__ = "invoice_reminders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), nullable=False)

    level: Mapped[int] = mapped_column(Integer, nullable=False)  # 1, 2, 3
    added_fee: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    generated_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    sent_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    pdf_url: Mapped[str] = mapped_column(String(500), nullable=True)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relations
    invoice = relationship("Invoice", back_populates="reminders")

    @override
    def __repr__(self):
        return f"<InvoiceReminder Level {self.level} - {self.added_fee} CHF>"

    def to_dict(self):
        """Sérialise le rappel en dictionnaire."""
        return {
            "id": self.id,
            "invoice_id": self.invoice_id,
            "level": self.level,
            "added_fee": float(self.added_fee),
            "generated_at": _iso(self.generated_at),
            "sent_at": _iso(self.sent_at),
            "pdf_url": self.pdf_url,
            "note": self.note,
        }


class CompanyBillingSettings(db.Model):
    """Paramètres de facturation par entreprise."""

    __tablename__ = "company_billing_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        ForeignKey("company.id"), nullable=False, unique=True
    )

    # Délais et frais
    payment_terms_days: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, default=10
    )
    overdue_fee: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2), nullable=True, default=15
    )
    reminder1_fee: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2), nullable=True, default=0
    )
    reminder2_fee: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2), nullable=True, default=40
    )
    reminder3_fee: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 2), nullable=True, default=0
    )
    vat_applicable = Column(Boolean, nullable=False, default=True)
    vat_rate: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 2), nullable=True, default=Decimal("7.7")
    )
    vat_label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    vat_number: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Planning des rappels (en jours)
    reminder_schedule_days = Column(
        JSON,
        nullable=False,
        default={
            "1": 10,  # 1er rappel 10j après échéance
            "2": 5,  # 2e rappel 5j après le 1er
            "3": 5,  # 3e rappel 5j après le 2e
        },
    )

    # Configuration
    auto_reminders_enabled = Column(Boolean, nullable=False, default=True)
    email_sender: Mapped[str] = mapped_column(String(200), nullable=True)

    # Format de numérotation
    invoice_number_format = Column(
        String(50), nullable=False, default="{PREFIX}-{YYYY}-{MM}-{SEQ4}"
    )
    invoice_prefix = Column(String(10), nullable=False, default="EM")

    # Informations bancaires
    iban: Mapped[str] = mapped_column(String(50), nullable=True)
    qr_iban: Mapped[str] = mapped_column(String(50), nullable=True)
    esr_ref_base: Mapped[str] = mapped_column(String(50), nullable=True)

    # Templates de messages
    invoice_message_template: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reminder1_template: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reminder2_template: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reminder3_template: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Pied de page légal
    legal_footer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pdf_template_variant = Column(String(20), nullable=False, default="default")

    # Relations
    company = relationship("Company", backref="billing_settings")

    @override
    def __repr__(self):
        return f"<CompanyBillingSettings {self.company_id}>"

    def to_dict(self):
        """Convertit l'objet en dictionnaire."""
        return {
            "id": self.id,
            "company_id": self.company_id,
            "payment_terms_days": self.payment_terms_days,
            "overdue_fee": float(self.overdue_fee)
            if self.overdue_fee is not None
            else None,
            "reminder1_fee": float(self.reminder1_fee)
            if self.reminder1_fee is not None
            else None,
            "reminder2_fee": float(self.reminder2_fee)
            if self.reminder2_fee is not None
            else None,
            "reminder3_fee": float(self.reminder3_fee)
            if self.reminder3_fee is not None
            else None,
            "reminder_schedule_days": self.reminder_schedule_days,
            "auto_reminders_enabled": self.auto_reminders_enabled,
            "email_sender": self.email_sender,
            "invoice_number_format": self.invoice_number_format,
            "invoice_prefix": self.invoice_prefix,
            "iban": self.iban,
            "qr_iban": self.qr_iban,
            "esr_ref_base": self.esr_ref_base,
            "invoice_message_template": self.invoice_message_template,
            "reminder1_template": self.reminder1_template,
            "reminder2_template": self.reminder2_template,
            "reminder3_template": self.reminder3_template,
            "legal_footer": self.legal_footer,
            "pdf_template_variant": self.pdf_template_variant,
            "vat_applicable": self.vat_applicable,
            "vat_rate": float(self.vat_rate) if self.vat_rate is not None else None,
            "vat_label": self.vat_label,
            "vat_number": self.vat_number,
        }


class InvoiceSequence(db.Model):
    """Séquence de numérotation des factures par entreprise et mois."""

    __tablename__ = "invoice_sequences"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("company.id"), nullable=False
    )
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    month: Mapped[int] = mapped_column(Integer, nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Relations
    company = relationship("Company", backref="invoice_sequences")

    # Contrainte d'unicité
    __table_args__ = (
        UniqueConstraint("company_id", "year", "month", name="uq_company_year_month"),
    )

    @override
    def __repr__(self):
        return f"<InvoiceSequence {self.company_id}-{self.year}-{self.month}: {self.sequence}>"
