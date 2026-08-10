"""Statuts dérivés des factures légales plateforme (registre)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from models.enums import PlatformIssuedDocumentType, PlatformIssuedInvoiceStatus
from models.platform_billing import PlatformIssuedInvoice
from services.platform_billing.money import money_round_chf

_TERMINAL = {
    PlatformIssuedInvoiceStatus.CANCELLED.value,
    PlatformIssuedInvoiceStatus.CREDITED.value,
    PlatformIssuedInvoiceStatus.PAID.value,
}


def document_type_of(inv: PlatformIssuedInvoice) -> str:
    return (
        getattr(inv, "document_type", None) or PlatformIssuedDocumentType.INVOICE.value
    )


def is_credit_note(inv: PlatformIssuedInvoice) -> bool:
    return document_type_of(inv) == PlatformIssuedDocumentType.CREDIT_NOTE.value


def balance_due_for_registry(inv: PlatformIssuedInvoice) -> Decimal:
    """Solde créance registre — 0 pour les avoirs."""
    if is_credit_note(inv):
        return Decimal("0.00")
    total = Decimal(str(inv.total_amount or 0))
    paid = Decimal(str(inv.amount_paid or 0))
    return money_round_chf(max(total - paid, Decimal("0.00")))


def credit_amount(inv: PlatformIssuedInvoice) -> Decimal | None:
    if not is_credit_note(inv):
        return None
    return money_round_chf(abs(Decimal(str(inv.total_amount or 0))))


def payment_state(inv: PlatformIssuedInvoice) -> str:
    if is_credit_note(inv):
        return "NONE"
    paid = money_round_chf(Decimal(str(inv.amount_paid or 0)))
    bal = balance_due_for_registry(inv)
    if paid <= 0:
        return "NONE"
    if bal <= 0:
        return "PAID"
    return "PARTIAL"


def is_overdue_read(inv: PlatformIssuedInvoice, *, now: datetime | None = None) -> bool:
    now = now or datetime.now(UTC)
    if is_credit_note(inv):
        return False
    if inv.status in _TERMINAL:
        return False
    if not inv.sent_at or not inv.due_at:
        return False
    if balance_due_for_registry(inv) <= 0:
        return False
    due = inv.due_at if inv.due_at.tzinfo else inv.due_at.replace(tzinfo=UTC)
    return due < now


def days_overdue(
    inv: PlatformIssuedInvoice, *, now: datetime | None = None
) -> int | None:
    now = now or datetime.now(UTC)
    if not is_overdue_read(inv, now=now) or not inv.due_at:
        return None
    due = inv.due_at if inv.due_at.tzinfo else inv.due_at.replace(tzinfo=UTC)
    return max(0, (now.date() - due.date()).days)


def ui_status(inv: PlatformIssuedInvoice, *, now: datetime | None = None) -> str:
    """Statut principal d'affichage (priorité plan)."""
    now = now or datetime.now(UTC)
    st = inv.status or PlatformIssuedInvoiceStatus.DRAFT.value
    if st == PlatformIssuedInvoiceStatus.CANCELLED.value:
        return "CANCELLED"
    if st == PlatformIssuedInvoiceStatus.CREDITED.value:
        return "CREDITED"
    if is_credit_note(inv):
        if inv.sent_at:
            return "SENT"
        return "ISSUED"
    bal = balance_due_for_registry(inv)
    if bal <= 0 and money_round_chf(Decimal(str(inv.amount_paid or 0))) > 0:
        return "PAID"
    if st == PlatformIssuedInvoiceStatus.PAID.value:
        return "PAID"
    if is_overdue_read(inv, now=now):
        return "OVERDUE"
    if payment_state(inv) == "PARTIAL":
        return "PARTIALLY_PAID"
    if inv.sent_at or st == PlatformIssuedInvoiceStatus.SENT.value:
        return "SENT"
    if st == PlatformIssuedInvoiceStatus.OVERDUE.value:
        return "OVERDUE"
    return "ISSUED"


def serialize_issued_status_fields(
    inv: PlatformIssuedInvoice, *, now: datetime | None = None
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    bal = balance_due_for_registry(inv)
    return {
        "document_type": document_type_of(inv),
        "balance_due": str(bal),
        "credit_amount": (
            str(credit_amount(inv)) if credit_amount(inv) is not None else None
        ),
        "payment_state": payment_state(inv),
        "is_overdue": is_overdue_read(inv, now=now),
        "days_overdue": days_overdue(inv, now=now),
        "ui_status": ui_status(inv, now=now),
    }
