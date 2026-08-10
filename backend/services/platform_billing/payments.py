"""Cycle de paiement des factures légales plateforme (ledger)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import func, select

from ext import db
from models.enums import (
    PlatformIssuedDocumentType,
    PlatformIssuedInvoiceStatus,
    PlatformPaymentEntryType,
)
from models.platform_billing import PlatformInvoicePayment, PlatformIssuedInvoice
from services.platform_billing.issued_status import (
    balance_due_for_registry,
    document_type_of,
    is_credit_note,
)
from services.platform_billing.money import money_round_chf

_ZURICH = ZoneInfo("Europe/Zurich")


_ALLOWED_CANCEL = {
    PlatformIssuedInvoiceStatus.DRAFT.value,
    PlatformIssuedInvoiceStatus.ISSUED.value,
}

_ALLOWED_CREDIT_SOURCE = {
    PlatformIssuedInvoiceStatus.ISSUED.value,
    PlatformIssuedInvoiceStatus.SENT.value,
    PlatformIssuedInvoiceStatus.OVERDUE.value,
}


def _as_utc(dt: datetime | None) -> datetime | None:
    """Normalise un datetime en UTC aware (évite naive vs aware)."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Naive : interpréter en Europe/Zurich (saisie admin locale), pas UTC.
        return dt.replace(tzinfo=_ZURICH).astimezone(UTC)
    return dt.astimezone(UTC)


def parse_payment_paid_at(raw: Any) -> datetime | None:
    """Parse ``paid_at`` API : date seule → jour local Zurich + heure courante si aujourd’hui.

    Une date ``YYYY-MM-DD`` ne doit pas devenir minuit UTC (affichage 02:00 en été).
    """
    if raw is None or raw == "":
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        day = date.fromisoformat(text)
        now_zh = datetime.now(_ZURICH)
        if day == now_zh.date():
            return now_zh
        return datetime(day.year, day.month, day.day, 12, 0, 0, tzinfo=_ZURICH)
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=_ZURICH)
    return dt


def _lock_invoice(issued_invoice_id: int) -> PlatformIssuedInvoice:
    inv = db.session.execute(
        select(PlatformIssuedInvoice)
        .where(PlatformIssuedInvoice.id == int(issued_invoice_id))
        .with_for_update()
    ).scalar_one_or_none()
    if inv is None:
        raise ValueError("Facture introuvable")
    return inv


def sum_ledger_amount(issued_invoice_id: int) -> Decimal:
    total = db.session.scalar(
        select(func.coalesce(func.sum(PlatformInvoicePayment.amount), 0)).where(
            PlatformInvoicePayment.issued_invoice_id == int(issued_invoice_id)
        )
    )
    return money_round_chf(Decimal(str(total or 0)))


def recompute_invoice_payment_state(
    invoice: PlatformIssuedInvoice, *, now: datetime | None = None
) -> None:
    """Recalcule amount_paid et le statut stocké depuis le journal."""
    now = _as_utc(now) or datetime.now(UTC)
    if is_credit_note(invoice):
        invoice.amount_paid = Decimal("0.00")
        return

    paid = sum_ledger_amount(invoice.id)
    invoice.amount_paid = paid
    total = money_round_chf(Decimal(str(invoice.total_amount or 0)))
    balance = money_round_chf(max(total - paid, Decimal("0.00")))

    if invoice.status in (
        PlatformIssuedInvoiceStatus.CANCELLED.value,
        PlatformIssuedInvoiceStatus.CREDITED.value,
    ):
        return

    if balance <= 0 and paid > 0:
        invoice.status = PlatformIssuedInvoiceStatus.PAID.value
        if invoice.paid_at is None:
            invoice.paid_at = now
        return

    invoice.paid_at = None
    if not invoice.sent_at:
        if invoice.status not in (
            PlatformIssuedInvoiceStatus.DRAFT.value,
            PlatformIssuedInvoiceStatus.ISSUED.value,
        ):
            invoice.status = PlatformIssuedInvoiceStatus.ISSUED.value
        return

    due = invoice.due_at
    if due is not None:
        due_aware = _as_utc(due)
        if due_aware is not None and due_aware < now and balance > 0:
            invoice.status = PlatformIssuedInvoiceStatus.OVERDUE.value
            return
    invoice.status = PlatformIssuedInvoiceStatus.SENT.value


def record_payment(
    issued_invoice_id: int,
    *,
    amount: Decimal,
    paid_at: datetime | None = None,
    method: str | None = None,
    reference: str | None = None,
    notes: str | None = None,
    created_by_user_id: int | None = None,
    idempotency_key: str | None = None,
) -> PlatformIssuedInvoice:
    inv = _lock_invoice(issued_invoice_id)
    if is_credit_note(inv):
        raise ValueError("Paiement impossible sur une note de crédit")
    if inv.status in (
        PlatformIssuedInvoiceStatus.CANCELLED.value,
        PlatformIssuedInvoiceStatus.CREDITED.value,
    ):
        raise ValueError("Facture annulée ou créditée")

    key = (idempotency_key or "").strip() or None
    if key:
        existing = PlatformInvoicePayment.query.filter_by(
            issued_invoice_id=inv.id, idempotency_key=key
        ).first()
        if existing is not None:
            db.session.refresh(inv)
            return inv

    amt = money_round_chf(amount)
    if amt <= 0:
        raise ValueError("Montant de paiement invalide")
    bal = balance_due_for_registry(inv)
    if amt > bal:
        raise ValueError(
            f"Paiement supérieur au solde ({bal} CHF) — trop-perçu interdit"
        )
    when = _as_utc(paid_at) or datetime.now(UTC)
    db.session.add(
        PlatformInvoicePayment(
            issued_invoice_id=inv.id,
            entry_type=PlatformPaymentEntryType.PAYMENT.value,
            amount=amt,
            paid_at=when,
            method=method,
            reference=reference,
            notes=notes,
            idempotency_key=key,
            created_by_user_id=created_by_user_id,
        )
    )
    db.session.flush()
    recompute_invoice_payment_state(inv, now=when)
    db.session.commit()
    db.session.refresh(inv)
    return inv


def reverse_payment(
    issued_invoice_id: int,
    payment_id: int,
    *,
    reason: str,
    created_by_user_id: int | None = None,
) -> PlatformIssuedInvoice:
    inv = _lock_invoice(issued_invoice_id)
    if is_credit_note(inv):
        raise ValueError("Contre-écriture impossible sur une note de crédit")
    if inv.status in (
        PlatformIssuedInvoiceStatus.CANCELLED.value,
        PlatformIssuedInvoiceStatus.CREDITED.value,
    ):
        raise ValueError("Facture annulée ou créditée")

    reason_clean = (reason or "").strip()
    if not reason_clean:
        raise ValueError("Motif de contre-écriture obligatoire")

    original = db.session.get(PlatformInvoicePayment, int(payment_id))
    if original is None or original.issued_invoice_id != inv.id:
        raise ValueError("Écriture de paiement introuvable")
    if original.entry_type != PlatformPaymentEntryType.PAYMENT.value:
        raise ValueError("Seule une écriture PAYMENT peut être contrepassée")
    if original.reverses_payment_id is not None:
        raise ValueError("Écriture invalide")

    already = PlatformInvoicePayment.query.filter_by(
        reverses_payment_id=original.id
    ).first()
    if already is not None:
        raise ValueError("Cette écriture a déjà été contrepassée")

    amt = money_round_chf(-Decimal(str(original.amount)))
    when = datetime.now(UTC)
    db.session.add(
        PlatformInvoicePayment(
            issued_invoice_id=inv.id,
            entry_type=PlatformPaymentEntryType.REVERSAL.value,
            amount=amt,
            paid_at=when,
            method=original.method,
            reference=original.reference,
            notes=original.notes,
            reverses_payment_id=original.id,
            reversal_reason=reason_clean[:512],
            created_by_user_id=created_by_user_id,
        )
    )
    db.session.flush()
    recompute_invoice_payment_state(inv, now=when)
    db.session.commit()
    db.session.refresh(inv)
    return inv


def mark_sent(issued_invoice_id: int) -> PlatformIssuedInvoice:
    inv = _lock_invoice(issued_invoice_id)
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
    inv = _lock_invoice(issued_invoice_id)
    if inv.status not in _ALLOWED_CANCEL:
        raise ValueError("Annulation impossible après envoi/paiement")
    if inv.status == PlatformIssuedInvoiceStatus.ISSUED.value and inv.sent_at:
        raise ValueError("Annulation impossible après envoi — créer une note de crédit")
    if money_round_chf(Decimal(str(inv.amount_paid or 0))) > 0:
        raise ValueError("Annulation impossible : des paiements sont enregistrés")
    inv.status = PlatformIssuedInvoiceStatus.CANCELLED.value
    inv.cancelled_at = datetime.now(UTC)
    db.session.commit()
    return inv


def _create_credit_note_no_commit(
    source: PlatformIssuedInvoice,
    *,
    reason: str,
    created_by_user_id: int | None = None,
) -> PlatformIssuedInvoice:
    """Crée un avoir total sans commit (pour transactions atomiques)."""
    reason_clean = (reason or "").strip()
    if not reason_clean:
        raise ValueError("Motif d'avoir obligatoire")
    if document_type_of(source) != PlatformIssuedDocumentType.INVOICE.value:
        raise ValueError("Seule une facture peut être créditée")
    if source.status not in _ALLOWED_CREDIT_SOURCE:
        raise ValueError("Note de crédit impossible pour ce statut")
    paid = money_round_chf(Decimal(str(source.amount_paid or 0)))
    if paid > 0:
        raise ValueError(
            "Avoir interdit dès qu'un paiement existe (remboursements hors périmètre)"
        )
    existing_cn = PlatformIssuedInvoice.query.filter_by(
        credit_of_invoice_id=source.id
    ).first()
    if existing_cn is not None:
        raise ValueError("Un avoir existe déjà pour cette facture")

    now = datetime.now(UTC)
    credit = PlatformIssuedInvoice(
        statement_id=None,
        company_id=source.company_id,
        invoice_number=f"{source.invoice_number}-AV-01",
        document_type=PlatformIssuedDocumentType.CREDIT_NOTE.value,
        status=PlatformIssuedInvoiceStatus.ISSUED.value,
        currency=source.currency,
        subtotal_amount=-source.subtotal_amount,
        tax_rate=source.tax_rate,
        tax_amount=-source.tax_amount,
        total_amount=-source.total_amount,
        qr_amount=Decimal("0.00"),
        qr_reference=None,
        issued_at=now,
        credit_of_invoice_id=source.id,
        billing_year=source.billing_year,
        billing_month=source.billing_month,
        period_id=source.period_id,
        credit_reason=reason_clean[:512],
        credit_created_by_user_id=created_by_user_id,
        debtor_snapshot=source.debtor_snapshot,
        creditor_snapshot=source.creditor_snapshot,
        lines_snapshot=source.lines_snapshot,
        billing_config_id=source.billing_config_id,
        partner_agreement_id=source.partner_agreement_id,
        dunning_policy_snapshot=source.dunning_policy_snapshot,
        dunning_automation_authorized_at_issuance=False,
        amount_paid=Decimal("0.00"),
    )
    source.status = PlatformIssuedInvoiceStatus.CREDITED.value
    source.credited_at = now
    db.session.add(credit)
    db.session.flush()

    try:
        from services.platform_billing.credit_note_pdf import (
            build_and_store_credit_note_pdf,
        )

        build_and_store_credit_note_pdf(credit, source)
    except Exception as exc:  # noqa: BLE001 — avoir DB valide même si PDF échoue
        import logging

        logging.getLogger(__name__).warning(
            "PDF avoir non généré pour %s: %s", credit.invoice_number, exc
        )
    return credit


def create_credit_note(
    issued_invoice_id: int,
    *,
    reason: str,
    created_by_user_id: int | None = None,
) -> PlatformIssuedInvoice:
    """Note de crédit totale — uniquement si aucun encaissement."""
    source = _lock_invoice(issued_invoice_id)
    credit = _create_credit_note_no_commit(
        source, reason=reason, created_by_user_id=created_by_user_id
    )
    db.session.commit()
    db.session.refresh(credit)
    return credit


def refresh_overdue_statuses(*, now: datetime | None = None) -> int:
    """Passe en OVERDUE les factures INVOICE envoyées échues avec solde restant."""
    now = now or datetime.now(UTC)
    rows = PlatformIssuedInvoice.query.filter(
        PlatformIssuedInvoice.document_type == PlatformIssuedDocumentType.INVOICE.value,
        PlatformIssuedInvoice.status.in_(
            [
                PlatformIssuedInvoiceStatus.SENT.value,
                PlatformIssuedInvoiceStatus.OVERDUE.value,
            ]
        ),
        PlatformIssuedInvoice.sent_at.isnot(None),
        PlatformIssuedInvoice.due_at.isnot(None),
        PlatformIssuedInvoice.due_at < now,
    ).all()
    changed = 0
    for inv in rows:
        recompute_invoice_payment_state(inv, now=now)
        if inv.status == PlatformIssuedInvoiceStatus.OVERDUE.value:
            changed += 1
    if changed:
        db.session.commit()
    return changed
