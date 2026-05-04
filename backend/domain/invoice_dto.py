"""DTO (Data Transfer Object) pour Invoice.

Ce DTO découple les services de l'implémentation SQLAlchemy.
Les services utilisent ce DTO au lieu d'accéder directement au modèle Invoice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from models.enums import InvoiceStatus


@dataclass(frozen=True, slots=True)
class InvoiceLineDTO:
    """DTO pour InvoiceLine - Sans dépendance SQLAlchemy."""

    id: int
    invoice_id: int
    line_type: str
    description: str
    quantity: Decimal
    unit_price: Decimal
    line_total: Decimal
    vat_rate: Decimal | None = None
    vat_amount: Decimal = Decimal("0.00")
    total_with_vat: Decimal = Decimal("0.00")
    adjustment_note: str | None = None
    reservation_id: int | None = None
    line_meta: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class InvoiceDTO:
    """DTO pour Invoice - Sans dépendance SQLAlchemy.

    Contient uniquement les champs essentiels utilisés par les services.
    Les champs optionnels sont marqués comme Optional.
    """

    # Identifiants
    id: int
    company_id: int
    client_id: int
    bill_to_client_id: int | None = None

    # Période de facturation
    period_month: int = 1
    period_year: int = 2024

    # Numéro de facture
    invoice_number: str = ""
    currency: str = "CHF"

    # Montants
    subtotal_amount: Decimal = Decimal("0.00")
    late_fee_amount: Decimal = Decimal("0.00")
    reminder_fee_amount: Decimal = Decimal("0.00")
    vat_total_amount: Decimal = Decimal("0.00")
    total_amount: Decimal = Decimal("0.00")
    amount_paid: Decimal = Decimal("0.00")
    balance_due: Decimal = Decimal("0.00")

    # Dates
    issued_at: datetime | None = None
    due_date: datetime | None = None
    sent_at: datetime | None = None
    paid_at: datetime | None = None
    updated_at: datetime | None = None

    # Statut
    status: InvoiceStatus = InvoiceStatus.DRAFT

    # Relations (chargées via eager loading si nécessaire)
    lines: list[InvoiceLineDTO] | None = None

    # Prévisualisation PDF / métadonnées (aligné sur models.Invoice.to_dict utilisateur)
    pdf_url: str | None = None
    meta: dict[str, Any] | None = None
    payments: list[dict[str, Any]] | None = None

    # Affichage brouillon / aperçu (GET détail avec lignes — alignement modèle Invoice)
    billing_strategy: str | None = None
    client: dict[str, Any] | None = None
    bill_to_client: dict[str, Any] | None = None
    billing_party: dict[str, Any] | None = None
    billed_to_company: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit le DTO en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "company_id": self.company_id,
            "client_id": self.client_id,
            "bill_to_client_id": self.bill_to_client_id,
            "billing_strategy": self.billing_strategy,
            "client": self.client,
            "bill_to_client": self.bill_to_client,
            "billing_party": self.billing_party,
            "billed_to_company": self.billed_to_company,
            "period_month": self.period_month,
            "period_year": self.period_year,
            "invoice_number": self.invoice_number,
            "currency": self.currency,
            "subtotal_amount": float(self.subtotal_amount),
            "late_fee_amount": float(self.late_fee_amount),
            "reminder_fee_amount": float(self.reminder_fee_amount),
            "vat_total_amount": float(self.vat_total_amount),
            "total_amount": float(self.total_amount),
            "amount_paid": float(self.amount_paid),
            "balance_due": float(self.balance_due),
            "issued_at": self.issued_at.isoformat() if self.issued_at else None,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "paid_at": self.paid_at.isoformat() if self.paid_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": (
                self.status.value if hasattr(self.status, "value") else str(self.status)
            ),
            "pdf_url": self.pdf_url,
            "meta": self.meta,
            "payments": list(self.payments) if self.payments is not None else [],
            "lines": [
                _invoice_line_dto_to_api_dict(line)
                for line in (self.lines or [])
            ],
        }


def _invoice_line_dto_to_api_dict(line: InvoiceLineDTO) -> dict[str, Any]:
    """Même forme que models.InvoiceLine.to_dict() pour le front (édition brouillon)."""
    d: dict[str, Any] = {
        "id": line.id,
        "invoice_id": line.invoice_id,
        "type": line.line_type,
        "description": line.description,
        "qty": float(line.quantity),
        "unit_price": float(line.unit_price),
        "line_total": float(line.line_total),
        "vat_rate": float(line.vat_rate) if line.vat_rate is not None else None,
        "vat_amount": float(line.vat_amount),
        "total_with_vat": float(line.total_with_vat),
        "adjustment_note": line.adjustment_note,
        "reservation_id": line.reservation_id,
    }
    if line.line_meta is not None:
        d["line_meta"] = line.line_meta
    return d
