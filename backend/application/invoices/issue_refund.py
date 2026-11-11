"""Restitution d'argent déjà encaissé.

Un avoir réduit la créance. Une correction rectifie une saisie. Ce service
n'écrit ni l'un ni l'autre : il enregistre une demande de remboursement, et
ne consomme le montant remboursable qu'à la confirmation de la remise.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from domain.end_of_trip_settlement.cash_settlement import money
from domain.end_of_trip_settlement.financial_contract import (
    CashLine,
    imputable,
    recognized_net,
)
from ext import db
from models import (
    CashCollection,
    CashCollectionEvent,
    CashHandover,
    Invoice,
    InvoiceCreditNote,
    InvoicePayment,
    InvoiceRefund,
)

logger = logging.getLogger(__name__)

_CENT = Decimal("0.01")
_OPEN = ("requested", "authorized")
_TERMINAL = ("refused", "cancelled")
_DECISIONS = {
    "authorize": "authorized",
    "refuse": "refused",
    "cancel": "cancelled",
}


class RefundRefusal(Exception):
    """Refus d'une restitution."""

    def __init__(self, code: str, status: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def _money(value: Any) -> Decimal:
    return money(value)


def _lock_invoice(invoice_id: int, company_id: int) -> Invoice | None:
    return db.session.scalar(
        select(Invoice)
        .where(Invoice.id == int(invoice_id), Invoice.company_id == int(company_id))
        .with_for_update()
    )


def _lock_collection(collection_id: int, company_id: int) -> CashCollection | None:
    return db.session.scalar(
        select(CashCollection)
        .where(
            CashCollection.id == int(collection_id),
            CashCollection.company_id == int(company_id),
        )
        .with_for_update()
    )


def _by_key(company_id: int, key: str) -> InvoiceRefund | None:
    return db.session.scalar(
        select(InvoiceRefund)
        .where(
            InvoiceRefund.company_id == int(company_id),
            InvoiceRefund.idempotency_key == key,
        )
        .with_for_update()
    )


def _same(
    row: InvoiceRefund,
    amount: Decimal,
    reason: str,
    invoice_id: int | None,
    collection_id: int | None,
) -> bool:
    stored_invoice = None if row.invoice_id is None else int(row.invoice_id)
    stored_collection = (
        None if row.cash_collection_id is None else int(row.cash_collection_id)
    )
    return (
        _money(row.amount_chf) == amount
        and row.reason == reason
        and stored_invoice == invoice_id
        and stored_collection == collection_id
    )


def _cash_room(
    origin: CashCollection, *, exclude_refund_id: int | None
) -> tuple[Decimal, Decimal]:
    """Encore restituable, et remises déjà confirmées.

    Lecture des collections et des remises déjà enregistrées. Le registre
    opérationnel de caisse n'est pas chargé. Aucun virement n'est exécuté.
    """
    zero = Decimal("0.00")
    if origin.kind != "collection" or origin.net_chf is None:
        return zero, zero
    if not imputable(
        state=origin.state,
        disposition=origin.disposition,
        net=Decimal(origin.net_chf),
        due=Decimal(origin.amount_due_chf),
    ):
        return zero, zero
    compensations = db.session.scalars(
        select(CashCollection.net_chf).where(
            CashCollection.compensates_collection_id == int(origin.id)
        )
    ).all()
    handed = db.session.scalars(
        select(CashHandover.amount_chf).where(
            CashHandover.collection_id == int(origin.id),
            CashHandover.state == "confirmed",
        )
    ).all()
    handed_total = sum((Decimal(amount) for amount in handed), zero).quantize(_CENT)
    held = (
        Decimal(origin.net_chf)
        + sum((Decimal(net) for net in compensations if net is not None), zero)
        - handed_total
    ).quantize(_CENT)
    pending = InvoiceRefund.query.filter(
        InvoiceRefund.cash_collection_id == int(origin.id),
        InvoiceRefund.status.in_(_OPEN),
    )
    if exclude_refund_id is not None:
        pending = pending.filter(InvoiceRefund.id != int(exclude_refund_id))
    reserved = sum((_money(row.amount_chf) for row in pending.all()), zero)
    return (held - reserved).quantize(_CENT), handed_total


def _refuse_if_cash_short(
    origin: CashCollection, amount: Decimal, *, exclude_refund_id: int | None
) -> None:
    room, handed = _cash_room(origin, exclude_refund_id=exclude_refund_id)
    if amount <= room:
        return
    if handed > 0 and amount <= (room + handed).quantize(_CENT):
        raise RefundRefusal("especes_deja_remises")
    raise RefundRefusal("depassement_autorise")


def _invoice_room(invoice: Invoice, *, exclude_refund_id: int | None) -> Decimal:
    invoice.update_balance()
    db.session.flush()
    room = _money(invoice.refundable_chf)
    pending = InvoiceRefund.query.filter(
        InvoiceRefund.invoice_id == int(invoice.id),
        InvoiceRefund.status.in_(_OPEN),
    )
    if exclude_refund_id is not None:
        pending = pending.filter(InvoiceRefund.id != int(exclude_refund_id))
    reserved = sum((_money(row.amount_chf) for row in pending.all()), Decimal("0.00"))
    return (room - reserved).quantize(_CENT)


def _linked_invoice_id(collection_id: int) -> int | None:
    payment = db.session.scalar(
        select(InvoicePayment)
        .where(InvoicePayment.cash_collection_id == int(collection_id))
        .limit(1)
    )
    if payment is None:
        return None
    return int(payment.invoice_id)


def _compensate(
    origin: CashCollection, amount: Decimal, key: str, actor_id: int, reason: str
) -> CashCollection:
    signed = (Decimal("0.00") - amount).quantize(_CENT)
    row = CashCollection(
        booking_id=int(origin.booking_id),
        company_id=int(origin.company_id),
        driver_id=int(origin.driver_id),
        debtor_label=origin.debtor_label,
        amount_due_chf=origin.amount_due_chf,
        currency="CHF",
        state="administrative",
        disposition="refund",
        decision_version=origin.decision_version,
        production_activation=False,
        legally_approved=False,
        change_confirmed=False,
        kind="compensation",
        compensates_collection_id=int(origin.id),
        client_operation_key=key,
        net_chf=signed,
        tendered_chf=amount,
        change_chf=Decimal("0.00"),
    )
    db.session.add(row)
    db.session.flush()
    for collection_id in (int(row.id), int(origin.id)):
        db.session.add(
            CashCollectionEvent(
                collection_id=collection_id,
                idempotency_key=key,
                action="confirm_cash_refund",
                from_state="administrative",
                to_state="administrative",
                actor_role="company",
                actor_id=int(actor_id),
                payload={
                    "nature": "remboursement",
                    "reason": reason,
                    "amount_chf": f"{amount:.2f}",
                    "compensates_collection_id": int(origin.id),
                },
            )
        )
    db.session.flush()
    return row


def _payload(
    row: InvoiceRefund,
    invoice: Invoice | None,
    *,
    replay: bool,
    recognized_net: Decimal | None,
) -> dict[str, Any]:
    note_count = 0
    if invoice is not None:
        note_count = InvoiceCreditNote.query.filter_by(invoice_id=invoice.id).count()
    return {
        "refund_id": int(row.id),
        "status": row.status,
        "amount_chf": float(_money(row.amount_chf)),
        "reason": row.reason,
        "nature": row.nature,
        "invoice_id": row.invoice_id,
        "cash_collection_id": row.cash_collection_id,
        "booking_id": row.booking_id,
        "confirmed_at": None
        if row.confirmed_at is None
        else row.confirmed_at.isoformat(),
        "replay": replay,
        "credit_notes_created": 0,
        "credit_notes_on_invoice": note_count,
        "amount_paid": None if invoice is None else float(_money(invoice.amount_paid)),
        "balance_due": None if invoice is None else float(_money(invoice.balance_due)),
        "refundable_chf": None
        if invoice is None
        else float(_money(invoice.refundable_chf)),
        "recognized_net_chf": None if recognized_net is None else float(recognized_net),
    }


def _recognized(booking_id: int) -> Decimal:
    """Net reconnu des lignes déjà enregistrées. N'écrit rien."""
    rows = db.session.scalars(
        select(CashCollection).where(CashCollection.booking_id == int(booking_id))
    ).all()
    lines: list[CashLine] = []
    for row in rows:
        if row.net_chf is None:
            continue
        lines.append(
            CashLine(
                kind=row.kind or "collection",
                state=row.state,
                disposition=row.disposition,
                net_chf=Decimal(row.net_chf),
                due_chf=Decimal(row.amount_due_chf),
            )
        )
    if not lines:
        return Decimal("0.00")
    return recognized_net(lines)


def _confirm(
    row: InvoiceRefund,
    invoice: Invoice | None,
    origin: CashCollection | None,
    *,
    actor_id: int,
) -> None:
    amount = _money(row.amount_chf)
    if invoice is not None:
        room = _invoice_room(invoice, exclude_refund_id=int(row.id))
        if room <= 0:
            raise RefundRefusal("rien_a_rembourser")
        if amount > room:
            raise RefundRefusal("depassement_remboursable")
    if origin is not None:
        if origin.kind != "collection" or origin.net_chf is None:
            raise RefundRefusal("origine_non_remboursable")
        _refuse_if_cash_short(origin, amount, exclude_refund_id=int(row.id))
        before = _money(origin.net_chf)
        _compensate(origin, amount, row.idempotency_key, actor_id, row.reason)
        db.session.refresh(origin)
        if _money(origin.net_chf) != before:
            raise RefundRefusal("origine_modifiee")
    row.status = "confirmed"
    row.confirmed_at = datetime.now(UTC)
    if invoice is not None:
        invoice.update_balance()


def _decision_status(decision: str | None, *, confirm: bool) -> str | None:
    choice = (decision or "").strip().lower()
    if not choice:
        return None
    if choice not in _DECISIONS:
        raise RefundRefusal("decision_inconnue", 400)
    if confirm:
        raise RefundRefusal("decision_ambigue", 400)
    return _DECISIONS[choice]


def _close_without_money(row: InvoiceRefund, status: str) -> None:
    """Passe un dossier ouvert vers un état sans restitution."""
    if status == "confirmed":
        raise RefundRefusal("confirmation_requise")
    row.status = status
    row.confirmed_at = None


def record_refund(
    *,
    company_id: int,
    created_by_user_id: int,
    amount_chf: Decimal,
    reason: str,
    idempotency_key: str,
    confirm: bool,
    decision: str | None = None,
    invoice_id: int | None = None,
    cash_collection_id: int | None = None,
) -> dict[str, Any]:
    """Demande, instruit ou confirme une restitution. N'émet aucun avoir."""
    key = (idempotency_key or "").strip()
    if not key:
        raise RefundRefusal("cle_idempotence_requise", 400)
    if len(key) > 80:
        raise RefundRefusal("cle_idempotence_trop_longue", 400)
    motive = (reason or "").strip()
    if len(motive) < 3:
        raise RefundRefusal("motif_requis", 400)
    administrative = _decision_status(decision, confirm=confirm)
    try:
        amount = _money(amount_chf)
    except ValueError as exc:
        raise RefundRefusal("montant_invalide", 400) from exc
    if amount <= 0:
        raise RefundRefusal("montant_invalide", 400)
    if invoice_id is None and cash_collection_id is None:
        raise RefundRefusal("cible_absente", 400)

    invoice: Invoice | None = None
    if invoice_id is not None:
        invoice = _lock_invoice(int(invoice_id), int(company_id))
        if invoice is None:
            raise RefundRefusal("facture_absente", 404)
        db.session.refresh(invoice)
        current = str(getattr(invoice.status, "value", invoice.status) or "").lower()
        if current in {"draft", "cancelled"}:
            raise RefundRefusal("facture_non_remboursable")
    origin: CashCollection | None = None
    if cash_collection_id is not None:
        origin = _lock_collection(int(cash_collection_id), int(company_id))
        if origin is None:
            raise RefundRefusal("collection_absente", 404)
        linked = _linked_invoice_id(int(origin.id))
        if linked is not None and invoice_id is None:
            raise RefundRefusal("facture_requise")
        if linked is not None and int(linked) != int(invoice_id or 0):
            raise RefundRefusal("facture_incoherente")
        if invoice is not None and int(origin.company_id) != int(invoice.company_id):
            raise RefundRefusal("autre_entreprise", 404)

    target_invoice = None if invoice_id is None else int(invoice_id)
    target_collection = None if cash_collection_id is None else int(cash_collection_id)
    existing = _by_key(company_id, key)
    if existing is not None:
        if not _same(existing, amount, motive, target_invoice, target_collection):
            raise RefundRefusal("conflit_idempotence")
        recognized = None if origin is None else _recognized(int(origin.booking_id))
        if existing.status == "confirmed":
            if confirm and administrative is None:
                replay = _payload(
                    existing, invoice, replay=True, recognized_net=recognized
                )
                db.session.rollback()
                return replay
            raise RefundRefusal("etat_clos")
        if administrative is not None:
            if existing.status == administrative:
                replay = _payload(
                    existing, invoice, replay=True, recognized_net=recognized
                )
                db.session.rollback()
                return replay
            if existing.status in _TERMINAL or (
                administrative == "authorized" and existing.status != "requested"
            ):
                raise RefundRefusal("etat_clos")
            _close_without_money(existing, administrative)
            db.session.commit()
            fresh_invoice = (
                None if invoice is None else db.session.get(Invoice, invoice.id)
            )
            return _payload(
                existing, fresh_invoice, replay=False, recognized_net=recognized
            )
        if not confirm or existing.status in _TERMINAL:
            if existing.status in _TERMINAL and confirm:
                raise RefundRefusal("etat_clos")
            replay = _payload(existing, invoice, replay=True, recognized_net=recognized)
            db.session.rollback()
            return replay
        _confirm(existing, invoice, origin, actor_id=created_by_user_id)
        db.session.commit()
        fresh_invoice = None if invoice is None else db.session.get(Invoice, invoice.id)
        recognized = None if origin is None else _recognized(int(origin.booking_id))
        return _payload(
            existing, fresh_invoice, replay=False, recognized_net=recognized
        )

    if invoice is not None:
        room = _invoice_room(invoice, exclude_refund_id=None)
        if room <= 0:
            raise RefundRefusal("rien_a_rembourser")
        if amount > room:
            raise RefundRefusal("depassement_remboursable")
    if origin is not None:
        if origin.kind != "collection" or origin.net_chf is None:
            raise RefundRefusal("origine_non_remboursable")
        _refuse_if_cash_short(origin, amount, exclude_refund_id=None)

    nature = "restitution_especes" if origin is not None else "restitution_facture"
    initial_status = administrative or "requested"
    row = InvoiceRefund(
        company_id=int(company_id),
        invoice_id=target_invoice,
        booking_id=None if origin is None else int(origin.booking_id),
        amount_chf=amount,
        reason=motive,
        nature=nature,
        status=initial_status,
        idempotency_key=key,
        cash_collection_id=target_collection,
        created_by_user_id=int(created_by_user_id),
        confirmed_at=None,
    )
    nested = db.session.begin_nested()
    db.session.add(row)
    try:
        db.session.flush()
    except IntegrityError as exc:
        nested.rollback()
        raced = _by_key(company_id, key)
        if raced is None or not _same(
            raced, amount, motive, target_invoice, target_collection
        ):
            raise RefundRefusal("conflit_idempotence") from exc
        recognized = None if origin is None else _recognized(int(origin.booking_id))
        replay = _payload(raced, invoice, replay=True, recognized_net=recognized)
        db.session.rollback()
        return replay
    else:
        nested.commit()

    if confirm:
        _confirm(row, invoice, origin, actor_id=created_by_user_id)
    db.session.commit()
    fresh_invoice = None if invoice is None else db.session.get(Invoice, invoice.id)
    recognized = None if origin is None else _recognized(int(origin.booking_id))
    stored = db.session.get(InvoiceRefund, row.id)
    if stored is None:
        raise RefundRefusal("remboursement_introuvable", 404)
    return _payload(stored, fresh_invoice, replay=False, recognized_net=recognized)
