"""Règlement enregistré après l'émission, sans réécrire la facture.

La clé de rejeu est la référence du paiement, persistée en base.
Un paiement manuel ne porte jamais de collection chauffeur.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ext import db
from models import Invoice, InvoicePayment
from models.enums import InvoiceStatus, PaymentMethod

_CENT = Decimal("0.01")


class PaymentRefusal(Exception):
    """Refus d'un règlement ultérieur."""

    def __init__(self, code: str, status: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def _money(value: Any) -> Decimal:
    return Decimal(str(value or 0)).quantize(_CENT)


def _moment(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).replace(microsecond=0)


def _content_differs(
    payment: InvoicePayment,
    amount: Decimal | None,
    method: str,
    paid_at: datetime,
    *,
    amount_given: bool,
    paid_at_given: bool,
) -> bool:
    """Vrai si la clé désigne déjà un autre montant, moyen ou date."""
    stored_method = str(getattr(payment.method, "value", payment.method))
    if stored_method != method:
        return True
    if amount_given and _money(payment.amount) != _money(amount):
        return True
    return bool(paid_at_given and _moment(payment.paid_at) != _moment(paid_at))


def _lock_invoice(invoice_id: int, company_id: int) -> Invoice | None:
    return db.session.scalar(
        select(Invoice)
        .where(Invoice.id == int(invoice_id), Invoice.company_id == int(company_id))
        .with_for_update()
    )


def _payment_by_reference(invoice_id: int, reference: str) -> InvoicePayment | None:
    return db.session.scalar(
        select(InvoicePayment)
        .where(
            InvoicePayment.invoice_id == int(invoice_id),
            InvoicePayment.reference == reference,
        )
        .with_for_update()
    )


def _result(
    invoice: Invoice, payment: InvoicePayment, *, replay: bool
) -> dict[str, Any]:
    return {
        "payment_id": int(payment.id),
        "reference": payment.reference,
        "amount": _money(payment.amount),
        "method": str(getattr(payment.method, "value", payment.method)),
        "paid_at": payment.paid_at,
        "cash_collection_id": payment.cash_collection_id,
        "amount_paid": _money(invoice.amount_paid),
        "balance_due": _money(invoice.balance_due),
        "status": str(getattr(invoice.status, "value", invoice.status)),
        "replay": replay,
    }


def _reference_conflict(exc: IntegrityError) -> bool:
    return "uq_invoice_payments_reference" in str(getattr(exc, "orig", exc))


def record_later_payment(
    *,
    company_id: int,
    invoice_id: int,
    amount: Decimal | None,
    method: str,
    paid_at: datetime,
    idempotency_key: str | None,
    cash_collection_id: int | None = None,
    amount_given: bool = True,
    paid_at_given: bool = True,
) -> dict[str, Any]:
    """Ajoute un paiement positif. Le total, les lignes et le PDF ne bougent pas."""
    if cash_collection_id is not None:
        raise PaymentRefusal("collection_interdite")
    if method not in PaymentMethod.choices():
        raise PaymentRefusal("moyen_invalide", 400)
    key = (idempotency_key or "").strip() or None
    if key is not None and len(key) > 100:
        raise PaymentRefusal("reference_trop_longue", 400)

    invoice = _lock_invoice(invoice_id, company_id)
    if invoice is None:
        raise PaymentRefusal("facture_absente", 404)
    db.session.refresh(invoice)
    status = str(getattr(invoice.status, "value", invoice.status) or "").lower()
    if status == InvoiceStatus.CANCELLED.value:
        raise PaymentRefusal("facture_annulee")

    if key is not None:
        existing = _payment_by_reference(invoice.id, key)
        if existing is not None:
            if _content_differs(
                existing,
                amount,
                method,
                paid_at,
                amount_given=amount_given,
                paid_at_given=paid_at_given,
            ):
                raise PaymentRefusal("conflit_idempotence")
            payload = _result(invoice, existing, replay=True)
            db.session.rollback()
            return payload

    balance = _money(invoice.balance_due)
    if balance <= 0:
        raise PaymentRefusal("facture_soldee")
    due = _money(amount) if amount_given and amount is not None else balance
    if due <= 0:
        raise PaymentRefusal("montant_invalide", 400)
    if due > balance:
        raise PaymentRefusal("depassement_solde")

    payment = InvoicePayment()
    payment.invoice_id = invoice.id
    payment.amount = due
    payment.method = PaymentMethod(method)
    payment.paid_at = paid_at
    payment.reference = key or f"pay-{uuid.uuid4().hex}"
    payment.cash_collection_id = None
    nested = db.session.begin_nested()
    db.session.add(payment)
    try:
        db.session.flush()
    except IntegrityError as exc:
        nested.rollback()
        if key is None or not _reference_conflict(exc):
            raise
        existing = _payment_by_reference(invoice.id, key)
        if existing is None or _content_differs(
            existing,
            due,
            method,
            paid_at,
            amount_given=True,
            paid_at_given=paid_at_given,
        ):
            raise PaymentRefusal("conflit_idempotence") from exc
        payload = _result(invoice, existing, replay=True)
        db.session.rollback()
        return payload
    else:
        nested.commit()

    db.session.expire(invoice, ["payments"])
    invoice.update_balance()
    settled = str(getattr(invoice.status, "value", invoice.status))
    if settled == InvoiceStatus.PAID.value:
        invoice.paid_at = paid_at
    db.session.commit()
    return _result(invoice, payment, replay=False)
