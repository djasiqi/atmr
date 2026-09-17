"""GET native d'une facture partenaire."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from application.invoices.partner_invoice_access import load_accessible_partner_invoice
from application.invoices.partner_invoice_serialize import (
    serialize_partner_invoice_detail,
)


@dataclass(frozen=True, slots=True)
class GetPartnerInvoiceResult:
    ok: bool
    invoice: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


def get_partner_invoice(
    *, company_id: int, partner_invoice_id: int
) -> GetPartnerInvoiceResult:
    """Charge partner_invoices.id — jamais invoices.id."""
    partner_invoice = load_accessible_partner_invoice(company_id, partner_invoice_id)
    if partner_invoice is None:
        return GetPartnerInvoiceResult(
            ok=False,
            error={"error": "Facture partenaire introuvable"},
            status_code=404,
        )
    return GetPartnerInvoiceResult(
        ok=True,
        invoice=serialize_partner_invoice_detail(
            partner_invoice, company_id=company_id
        ),
    )
