"""Paiement natif rattaché à partner_invoices."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from application.invoices.partner_invoice_access import load_accessible_partner_invoice
from application.invoices.partner_invoice_serialize import (
    serialize_partner_invoice_detail,
)
from ext import db
from infrastructure.invoices.invoice_calculator import round_to_5_cents
from models.partner_invoice import PartnerInvoicePayment, PartnerInvoiceStatus


@dataclass(frozen=True, slots=True)
class AddPartnerInvoicePaymentResult:
    ok: bool
    invoice: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


def _parse_paid_at(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if value:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.now(UTC)


def add_partner_invoice_payment(
    *,
    company_id: int,
    partner_invoice_id: int,
    payload: dict[str, Any],
) -> AddPartnerInvoicePaymentResult:
    """Enregistre un paiement sur partner:id — jamais invoices.id."""
    partner_invoice = load_accessible_partner_invoice(company_id, partner_invoice_id)
    if partner_invoice is None:
        return AddPartnerInvoicePaymentResult(
            ok=False,
            error={"error": "Facture partenaire introuvable"},
            status_code=404,
        )
    if partner_invoice.status in {
        PartnerInvoiceStatus.CANCELLED,
        PartnerInvoiceStatus.PAID,
    }:
        return AddPartnerInvoicePaymentResult(
            ok=False,
            error={"error": "Cette facture partenaire n'accepte plus de paiement."},
            status_code=400,
        )

    try:
        amount = round_to_5_cents(Decimal(str(payload.get("amount"))))
    except Exception:
        return AddPartnerInvoicePaymentResult(
            ok=False,
            error={"error": "Montant de paiement invalide."},
            status_code=400,
        )
    if amount <= 0:
        return AddPartnerInvoicePaymentResult(
            ok=False,
            error={"error": "Le montant doit être supérieur à zéro."},
            status_code=400,
        )

    total = Decimal(str(partner_invoice.total_amount or 0))
    already_paid = Decimal(str(partner_invoice.amount_paid or 0))
    balance = total - already_paid
    if amount > balance:
        return AddPartnerInvoicePaymentResult(
            ok=False,
            error={"error": "Le montant dépasse le solde dû."},
            status_code=400,
        )

    method = str(payload.get("method") or "bank_transfer")[:50]
    note = payload.get("note")
    payment = PartnerInvoicePayment(
        partner_invoice_id=partner_invoice.id,
        amount=amount,
        method=method,
        paid_at=_parse_paid_at(payload.get("paid_at")),
        note=str(note)[:1000] if note else None,
    )
    db.session.add(payment)

    partner_invoice.amount_paid = already_paid + amount
    if partner_invoice.amount_paid >= total:
        partner_invoice.status = PartnerInvoiceStatus.PAID
        partner_invoice.paid_at = payment.paid_at
    else:
        partner_invoice.status = PartnerInvoiceStatus.PARTIALLY_PAID

    db.session.commit()
    refreshed = load_accessible_partner_invoice(company_id, partner_invoice_id)
    assert refreshed is not None
    return AddPartnerInvoicePaymentResult(
        ok=True,
        invoice=serialize_partner_invoice_detail(refreshed, company_id=company_id),
    )
