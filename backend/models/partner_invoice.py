# models/partner_invoice.py
"""Modèle PartnerInvoice - Facturation mensuelle consolidée des partenaires."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    func,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from ext import db


class PartnerInvoiceStatus:
    """Statuts des factures partenaires."""

    DRAFT = "draft"  # Brouillon (en cours de génération)
    SENT = "sent"  # Envoyée au partenaire
    PARTIALLY_PAID = "partially_paid"  # Partiellement payée
    PAID = "paid"  # Payée
    OVERDUE = "overdue"  # En retard
    CANCELLED = "cancelled"  # Annulée


class PartnerInvoice(db.Model):
    """Facture mensuelle consolidée pour un partenaire."""

    __tablename__ = "partner_invoices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    partnership_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("partnerships.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Entreprise exécutante (qui a effectué les transferts)
    # Permet d'avoir deux factures pour la même période si les entreprises exécutantes sont différentes
    executing_company_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("company.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Période facturée
    period_year: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    period_month: Mapped[int] = mapped_column(Integer, nullable=False, index=True)

    # Numéro de facture
    invoice_number: Mapped[str] = mapped_column(
        String(100), nullable=False, unique=True
    )

    # Montants
    subtotal_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    vat_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    total_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )
    amount_paid: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )  # Montant payé (pour les paiements partiels)
    credit_balance: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )  # Crédit disponible à déduire de la prochaine facture
    tip_amount: Mapped[Decimal] = mapped_column(
        Numeric(10, 2), nullable=False, default=0
    )  # Pourboire (ne sera pas déduit de la prochaine facture)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="CHF")

    # Statut
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=PartnerInvoiceStatus.DRAFT
    )

    # Dates
    issued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    due_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    paid_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Métadonnées
    pdf_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    notes: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Relations
    partnership = relationship("Partnership", backref="invoices")
    transfers = relationship(
        "BookingTransfer",
        secondary="partner_invoice_transfers",
        backref="partner_invoices",
    )

    # Pas de contrainte unique sur (partnership_id, period_year, period_month, executing_company_id)
    # car on peut avoir plusieurs factures pour la même période si ce sont des transferts différents
    # La seule contrainte est que chaque transfert ne peut être facturé qu'une seule fois
    # (gérée au niveau de la table de liaison partner_invoice_transfers)

    def to_dict(self) -> dict[str, Any]:
        """Sérialise la facture partenaire en dictionnaire."""
        return {
            "id": self.id,
            "partnership_id": self.partnership_id,
            "executing_company_id": self.executing_company_id,
            "period_year": self.period_year,
            "period_month": self.period_month,
            "invoice_number": self.invoice_number,
            "subtotal_amount": float(self.subtotal_amount),
            "vat_amount": float(self.vat_amount),
            "total_amount": float(self.total_amount),
            "amount_paid": float(self.amount_paid),
            "credit_balance": float(self.credit_balance),
            "tip_amount": float(self.tip_amount),
            "currency": self.currency,
            "status": self.status,
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "pdf_url": self.pdf_url,
            "notes": self.notes,
            "partnership": self.partnership.to_dict() if self.partnership else None,
            "transfers_count": len(self.transfers) if self.transfers else 0,
        }


# Table de liaison entre PartnerInvoice et BookingTransfer
partner_invoice_transfers = db.Table(
    "partner_invoice_transfers",
    db.Column(
        "partner_invoice_id",
        Integer,
        ForeignKey("partner_invoices.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    db.Column(
        "booking_transfer_id",
        Integer,
        ForeignKey("booking_transfers.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)
