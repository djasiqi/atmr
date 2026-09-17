"""PATCH native d'une facture partenaire (brouillon / envoyée non soldée)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from application.invoices.partner_invoice_access import load_accessible_partner_invoice
from application.invoices.partner_invoice_lines import (
    apply_partner_invoice_line_updates,
    ensure_partner_invoice_lines,
    recalculate_partner_invoice_totals,
)
from application.invoices.partner_invoice_serialize import (
    serialize_partner_invoice_detail,
)
from ext import db
from models.partner_invoice import PartnerInvoiceStatus

_EDITABLE_STATUSES = {
    PartnerInvoiceStatus.DRAFT,
    PartnerInvoiceStatus.SENT,
    PartnerInvoiceStatus.PARTIALLY_PAID,
    PartnerInvoiceStatus.OVERDUE,
}


def _parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class UpdatePartnerInvoiceResult:
    ok: bool
    invoice: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


def update_partner_invoice_draft(
    *,
    company_id: int,
    partner_invoice_id: int,
    payload: dict[str, Any],
) -> UpdatePartnerInvoiceResult:
    """Met à jour le snapshot partenaire. Ne réécrit pas les transferts source."""
    partner_invoice = load_accessible_partner_invoice(company_id, partner_invoice_id)
    if partner_invoice is None:
        return UpdatePartnerInvoiceResult(
            ok=False,
            error={"error": "Facture partenaire introuvable"},
            status_code=404,
        )
    if partner_invoice.status not in _EDITABLE_STATUSES:
        return UpdatePartnerInvoiceResult(
            ok=False,
            error={
                "error": (
                    "Cette facture partenaire ne peut plus être modifiée "
                    f"(statut: {partner_invoice.status})."
                )
            },
            status_code=400,
        )

    ensure_partner_invoice_lines(partner_invoice)

    if "notes" in payload:
        notes = payload.get("notes")
        partner_invoice.notes = str(notes)[:1000] if notes else None
    if "recipient_name" in payload:
        name = payload.get("recipient_name")
        partner_invoice.recipient_name = str(name)[:200] if name else None
    if "recipient_address" in payload:
        addr = payload.get("recipient_address")
        partner_invoice.recipient_address = str(addr)[:1000] if addr else None
    if "recipient_contact" in payload:
        contact = payload.get("recipient_contact")
        partner_invoice.recipient_contact = str(contact)[:300] if contact else None
    if "period_year" in payload and payload["period_year"] is not None:
        partner_invoice.period_year = int(payload["period_year"])
    if "period_month" in payload and payload["period_month"] is not None:
        month = int(payload["period_month"])
        if month < 1 or month > 12:
            return UpdatePartnerInvoiceResult(
                ok=False,
                error={"error": "Mois de période invalide"},
                status_code=400,
            )
        partner_invoice.period_month = month
    if "issued_at" in payload:
        issued = _parse_datetime(payload.get("issued_at"))
        if issued is not None:
            partner_invoice.issued_at = issued
    if "due_date" in payload:
        due = _parse_datetime(payload.get("due_date"))
        if due is not None:
            partner_invoice.due_date = due

    lines_payload = payload.get("lines")
    if isinstance(lines_payload, list):
        apply_partner_invoice_line_updates(partner_invoice, lines_payload)
        recalculate_partner_invoice_totals(partner_invoice)

    db.session.commit()
    refreshed = load_accessible_partner_invoice(company_id, partner_invoice_id)
    assert refreshed is not None
    return UpdatePartnerInvoiceResult(
        ok=True,
        invoice=serialize_partner_invoice_detail(refreshed, company_id=company_id),
    )
