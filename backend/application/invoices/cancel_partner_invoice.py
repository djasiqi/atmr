"""Annulation native d'une facture partenaire."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from application.invoices.partner_invoice_access import load_accessible_partner_invoice
from application.invoices.partner_invoice_serialize import (
    serialize_partner_invoice_detail,
)
from ext import db
from models.partner_invoice import PartnerInvoiceStatus


@dataclass(frozen=True, slots=True)
class CancelPartnerInvoiceResult:
    ok: bool
    invoice: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


def cancel_partner_invoice(
    *, company_id: int, partner_invoice_id: int
) -> CancelPartnerInvoiceResult:
    """Annule partner_invoices.id. N'écrit jamais invoices.id."""
    partner_invoice = load_accessible_partner_invoice(company_id, partner_invoice_id)
    if partner_invoice is None:
        return CancelPartnerInvoiceResult(
            ok=False,
            error={"error": "Facture partenaire introuvable"},
            status_code=404,
        )
    if partner_invoice.status == PartnerInvoiceStatus.CANCELLED:
        return CancelPartnerInvoiceResult(
            ok=False,
            error={"error": "Cette facture partenaire est déjà annulée."},
            status_code=400,
        )
    if partner_invoice.status == PartnerInvoiceStatus.PAID:
        return CancelPartnerInvoiceResult(
            ok=False,
            error={"error": "Une facture partenaire payée ne peut pas être annulée."},
            status_code=400,
        )
    if Decimal(str(partner_invoice.amount_paid or 0)) > 0:
        return CancelPartnerInvoiceResult(
            ok=False,
            error={
                "error": "Impossible d'annuler une facture partenaire déjà encaissée."
            },
            status_code=400,
        )
    if partner_invoice.status not in {
        PartnerInvoiceStatus.DRAFT,
        PartnerInvoiceStatus.SENT,
        PartnerInvoiceStatus.OVERDUE,
    }:
        return CancelPartnerInvoiceResult(
            ok=False,
            error={
                "error": (
                    f"Annulation refusée pour le statut {partner_invoice.status}."
                )
            },
            status_code=400,
        )

    partner_invoice.status = PartnerInvoiceStatus.CANCELLED
    db.session.commit()
    refreshed = load_accessible_partner_invoice(company_id, partner_invoice_id)
    assert refreshed is not None
    return CancelPartnerInvoiceResult(
        ok=True,
        invoice=serialize_partner_invoice_detail(refreshed, company_id=company_id),
    )
