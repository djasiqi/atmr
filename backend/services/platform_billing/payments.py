"""Cycle de paiement des factures légales plateforme."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from ext import db
from models.enums import PlatformIssuedInvoiceStatus
from models.platform_billing import PlatformInvoicePayment, PlatformIssuedInvoice
from services.platform_billing.money import money_round_chf


_ALLOWED_CANCEL = {
    PlatformIssuedInvoiceStatus.DRAFT.value,
    PlatformIssuedInvoiceStatus.ISSUED.value,
}


def record_payment(
    issued_invoice_id: int,
    *,
    amount: Decimal,
    paid_at: datetime | None = None,
    method: str | None = None,
    reference: str | None = None,
    notes: str | None = None,
    created_by_user_id: int | None = None,
) -> PlatformIssuedInvoice:
    inv = db.session.get(PlatformIssuedInvoice, issued_invoice_id)
    if not inv:
        raise ValueError("Facture introuvable")
    if inv.status in (
        PlatformIssuedInvoiceStatus.CANCELLED.value,
        PlatformIssuedInvoiceStatus.CREDITED.value,
    ):
        raise ValueError("Facture annulée ou créditée")
    amt = money_round_chf(amount)
    if amt <= 0:
        raise ValueError("Montant de paiement invalide")
    when = paid_at or datetime.now(UTC)
    db.session.add(
        PlatformInvoicePayment(
            issued_invoice_id=inv.id,
            amount=amt,
            paid_at=when,
            method=method,
            reference=reference,
            notes=notes,
            created_by_user_id=created_by_user_id,
        )
    )
    inv.amount_paid = money_round_chf(Decimal(str(inv.amount_paid or 0)) + amt)
    if inv.amount_paid >= inv.total_amount:
        inv.status = PlatformIssuedInvoiceStatus.PAID.value
        inv.paid_at = when
    elif inv.status == PlatformIssuedInvoiceStatus.OVERDUE.value:
        inv.status = PlatformIssuedInvoiceStatus.SENT.value
    db.session.commit()
    db.session.refresh(inv)
    return inv


def mark_sent(issued_invoice_id: int) -> PlatformIssuedInvoice:
    inv = db.session.get(PlatformIssuedInvoice, issued_invoice_id)
    if not inv:
        raise ValueError("Facture introuvable")
    if inv.status not in (
        PlatformIssuedInvoiceStatus.ISSUED.value,
        PlatformIssuedInvoiceStatus.DRAFT.value,
    ):
        raise ValueError(f"Transition envoi interdite depuis {inv.status}")
    inv.status = PlatformIssuedInvoiceStatus.SENT.value
    inv.sent_at = datetime.now(UTC)
    db.session.commit()
    return inv


def cancel_issued_invoice(issued_invoice_id: int) -> PlatformIssuedInvoice:
    inv = db.session.get(PlatformIssuedInvoice, issued_invoice_id)
    if not inv:
        raise ValueError("Facture introuvable")
    if inv.status not in _ALLOWED_CANCEL:
        raise ValueError("Annulation impossible après envoi/paiement")
    if inv.status == PlatformIssuedInvoiceStatus.ISSUED.value and inv.sent_at:
        raise ValueError("Annulation impossible après envoi — créer une note de crédit")
    inv.status = PlatformIssuedInvoiceStatus.CANCELLED.value
    inv.cancelled_at = datetime.now(UTC)
    db.session.commit()
    return inv


def create_credit_note(issued_invoice_id: int) -> PlatformIssuedInvoice:
    """Note de crédit après émission (nouvelle facture liée)."""
    source = db.session.get(PlatformIssuedInvoice, issued_invoice_id)
    if not source:
        raise ValueError("Facture introuvable")
    if source.status in (
        PlatformIssuedInvoiceStatus.CANCELLED.value,
        PlatformIssuedInvoiceStatus.CREDITED.value,
        PlatformIssuedInvoiceStatus.DRAFT.value,
    ):
        raise ValueError("Note de crédit impossible pour ce statut")
    credit = PlatformIssuedInvoice(
        statement_id=None,  # évite UNIQUE(statement_id) de la facture primaire
        company_id=source.company_id,
        invoice_number=f"{source.invoice_number}-CN",
        status=PlatformIssuedInvoiceStatus.CREDITED.value,
        currency=source.currency,
        subtotal_amount=-source.subtotal_amount,
        tax_rate=source.tax_rate,
        tax_amount=-source.tax_amount,
        total_amount=-source.total_amount,
        qr_amount=Decimal("0.00"),
        qr_reference=None,
        issued_at=datetime.now(UTC),
        credited_at=datetime.now(UTC),
        credit_of_invoice_id=source.id,
        debtor_snapshot=source.debtor_snapshot,
        creditor_snapshot=source.creditor_snapshot,
    )
    source.status = PlatformIssuedInvoiceStatus.CREDITED.value
    source.credited_at = datetime.now(UTC)
    db.session.add(credit)
    db.session.commit()
    db.session.refresh(credit)
    return credit


def refresh_overdue_statuses(*, now: datetime | None = None) -> int:
    now = now or datetime.now(UTC)
    rows = PlatformIssuedInvoice.query.filter(
        PlatformIssuedInvoice.status.in_(
            [
                PlatformIssuedInvoiceStatus.ISSUED.value,
                PlatformIssuedInvoiceStatus.SENT.value,
            ]
        ),
        PlatformIssuedInvoice.due_at.isnot(None),
        PlatformIssuedInvoice.due_at < now,
    ).all()
    for inv in rows:
        inv.status = PlatformIssuedInvoiceStatus.OVERDUE.value
    if rows:
        db.session.commit()
    return len(rows)
