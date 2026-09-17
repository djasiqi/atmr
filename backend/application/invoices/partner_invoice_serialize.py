"""Sérialisation détail d'une facture partenaire (catalogue isolé)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from application.invoices.partner_invoice_lines import serialize_partner_invoice_lines
from models.partner_invoice import PartnerInvoice, PartnerInvoiceStatus


def _d(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    return Decimal(str(value))


def billed_company_name(partner_invoice: PartnerInvoice, company_id: int) -> str:
    """Nom du destinataire affiché (partenaire facturé)."""
    if partner_invoice.recipient_name:
        return partner_invoice.recipient_name
    partnership = partner_invoice.partnership
    if not partnership:
        return "Entreprise partenaire"
    if partnership.owner_company_id == company_id and partnership.partner_company:
        return partnership.partner_company.name or "Entreprise partenaire"
    if partnership.owner_company:
        return partnership.owner_company.name or "Entreprise partenaire"
    return "Entreprise partenaire"


def serialize_partner_invoice_detail(
    partner_invoice: PartnerInvoice, *, company_id: int
) -> dict[str, Any]:
    """Dict détail pour GET/PATCH partenaire."""
    amount_paid = _d(partner_invoice.amount_paid)
    total_amount = _d(partner_invoice.total_amount)
    balance_due = total_amount - amount_paid
    if partner_invoice.status == PartnerInvoiceStatus.CANCELLED:
        effective_status = partner_invoice.status
    elif balance_due <= 0 and total_amount > 0:
        effective_status = PartnerInvoiceStatus.PAID
    elif amount_paid > 0:
        effective_status = PartnerInvoiceStatus.PARTIALLY_PAID
    else:
        effective_status = partner_invoice.status

    partner_name = billed_company_name(partner_invoice, company_id)
    payments = [
        payment.to_dict() for payment in (partner_invoice.recorded_payments or [])
    ]
    return {
        "id": partner_invoice.id,
        "invoice_number": partner_invoice.invoice_number,
        "period_year": partner_invoice.period_year,
        "period_month": partner_invoice.period_month,
        "subtotal_amount": float(_d(partner_invoice.subtotal_amount)),
        "vat_amount": float(_d(partner_invoice.vat_amount)),
        "total_amount": float(total_amount),
        "amount_paid": float(amount_paid),
        "balance_due": float(balance_due),
        "credit_balance": float(_d(partner_invoice.credit_balance)),
        "currency": partner_invoice.currency,
        "status": effective_status,
        "issued_at": (
            partner_invoice.issued_at.isoformat() if partner_invoice.issued_at else None
        ),
        "due_date": (
            partner_invoice.due_date.isoformat() if partner_invoice.due_date else None
        ),
        "paid_at": (
            partner_invoice.paid_at.isoformat() if partner_invoice.paid_at else None
        ),
        "sent_at": (
            partner_invoice.sent_at.isoformat() if partner_invoice.sent_at else None
        ),
        "pdf_url": partner_invoice.pdf_url,
        "notes": partner_invoice.notes,
        "recipient_name": partner_invoice.recipient_name or partner_name,
        "recipient_address": partner_invoice.recipient_address,
        "recipient_contact": partner_invoice.recipient_contact,
        "client": {
            "id": None,
            "first_name": "",
            "last_name": "",
            "username": "",
            "is_institution": True,
            "institution_name": partner_name,
        },
        "bill_to_client": None,
        "lines": serialize_partner_invoice_lines(partner_invoice),
        "payments": payments,
        "reminders": [],
        "reminder_level": 0,
        "last_reminder_at": None,
        "is_partner_invoice": True,
        "kind": "partner",
        "invoice_type": "partner",
        "partnership_id": partner_invoice.partnership_id,
        "company_id": partner_invoice.executing_company_id,
    }
