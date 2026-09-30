"""Factures transporteur visibles par le client débiteur.

Le client voit les factures qui lui ont été émises : envoyées, partiellement
payées, échues ou payées. Les brouillons et les factures annulées restent
invisibles. Le solde est celui de la facture, jamais le montant d'une course.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import joinedload

from models.client import Client
from models.enums import InvoiceStatus
from models.invoice import Invoice

_VISIBLE_STATUSES = (
    InvoiceStatus.SENT,
    InvoiceStatus.PARTIALLY_PAID,
    InvoiceStatus.PAID,
    InvoiceStatus.OVERDUE,
)


def _status_value(status: Any) -> str:
    raw = getattr(status, "value", status)
    return str(raw or "").strip().lower()


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def serialize_company_invoice_for_client(invoice: Invoice) -> dict[str, Any]:
    status = _status_value(invoice.status)
    balance = float(invoice.balance_due or 0)
    due = invoice.due_date
    now = datetime.now(UTC)
    due_aware = due
    if due is not None and due.tzinfo is None:
        due_aware = due.replace(tzinfo=UTC)
    is_overdue = status == InvoiceStatus.OVERDUE.value or (
        due_aware is not None and balance > 0 and due_aware < now
    )
    company = getattr(invoice, "company", None)
    company_name = str(getattr(company, "name", "") or "").strip() or "Transporteur"
    return {
        "receivable_id": f"invoice-{invoice.id}",
        "source": "company_invoice",
        "invoice_id": int(invoice.id),
        "creditor_company_name": company_name,
        "external_invoice_number": invoice.invoice_number,
        "issued_at": _iso(invoice.issued_at),
        "due_date": _iso(invoice.due_date),
        "currency": invoice.currency or "CHF",
        "total_amount": float(invoice.total_amount or 0),
        "amount_paid": float(invoice.amount_paid or 0),
        "balance_due": balance,
        "status": status or InvoiceStatus.SENT.value,
        "is_overdue": bool(is_overdue and status != InvoiceStatus.PAID.value),
        "hold_effect": None,
        "can_dispute": False,
        "dispute_status": None,
        "dunning_history": [],
        "pdf_available": bool(str(invoice.pdf_url or "").strip()),
    }


def client_ids_for_user(user_id: int) -> list[int]:
    return [int(row.id) for row in Client.query.filter_by(user_id=int(user_id)).all()]


def company_invoice_visible_to_client(user_id: int, invoice_id: int) -> Invoice | None:
    """Facture émise appartenant au client. Brouillons et annulées restent inaccessibles."""
    client_ids = client_ids_for_user(user_id)
    if not client_ids:
        return None
    return (
        Invoice.query.options(joinedload(Invoice.company))
        .filter(
            Invoice.id == int(invoice_id),
            or_(
                Invoice.client_id.in_(client_ids),
                Invoice.bill_to_client_id.in_(client_ids),
            ),
            Invoice.status.in_(_VISIBLE_STATUSES),
        )
        .one_or_none()
    )


def list_company_invoices_for_client_user(user_id: int) -> list[dict[str, Any]]:
    client_ids = client_ids_for_user(user_id)
    if not client_ids:
        return []
    invoices = (
        Invoice.query.options(joinedload(Invoice.company))
        .filter(
            or_(
                Invoice.client_id.in_(client_ids),
                Invoice.bill_to_client_id.in_(client_ids),
            ),
            Invoice.status.in_(_VISIBLE_STATUSES),
        )
        .order_by(Invoice.issued_at.desc().nulls_last(), Invoice.id.desc())
        .all()
    )
    return [serialize_company_invoice_for_client(invoice) for invoice in invoices]
