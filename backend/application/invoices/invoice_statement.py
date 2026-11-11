"""Situation financière d'une facture déjà émise.

Lecture seule. Les montants viennent de ``settle`` et des lignes
enregistrées. Ce module n'écrit ni paiement, ni avoir, ni PDF.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from domain.end_of_trip_settlement.financial_contract import (
    KIND_HANDOVER,
    credit_room,
    money,
    settle,
)

_UNAVAILABLE = "Situation financière indisponible."
_MISMATCH = (
    "Les montants enregistrés ne concordent pas avec le détail de la facture."
)

_MONEY_KEYS = (
    "prestations_chf",
    "avoirs_chf",
    "net_facture_chf",
    "acomptes_chf",
    "paiements_complementaires_chf",
    "remboursements_confirmes_chf",
    "reste_a_payer_chf",
    "a_rembourser_chf",
)

_RANK = {
    "emission": 0,
    "acompte": 1,
    "paiement": 2,
    "avoir": 3,
    "remboursement": 4,
}


def _plain(amount: Decimal) -> str:
    return f"{money(amount):.2f}"


def _reduction(amount: Decimal) -> str:
    quantized = money(amount)
    if quantized == 0:
        return "0.00"
    return f"-{quantized:.2f}"


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso(value: datetime | None) -> str | None:
    stamp = _as_utc(value)
    if stamp is None:
        return None
    return stamp.isoformat()


def _blank_amounts() -> dict[str, None]:
    return dict.fromkeys(_MONEY_KEYS)


def _unavailable(message: str, documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = _blank_amounts()
    payload.update(
        {
            "disponible": False,
            "message": message,
            "events": [],
            "documents": documents or [],
        }
    )
    return payload


def _documents(
    invoice_number: str | None,
    pdf_url: str | None,
    credit_notes: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    documents = [
        {
            "label": "Facture originale",
            "reference": invoice_number,
            "pdf_url": pdf_url or None,
            "disponible": bool(pdf_url),
        }
    ]
    for note in credit_notes or []:
        url = note.get("pdf_url") or None
        documents.append(
            {
                "label": "Avoir",
                "reference": note.get("number"),
                "pdf_url": url,
                "disponible": bool(url),
            }
        )
    return documents


def _event(
    *,
    at: datetime | None,
    rank: str,
    label: str,
    reference: str | None,
    montant: str | None,
    sequence: int,
) -> dict[str, Any]:
    return {
        "_at": _as_utc(at),
        "_rank": _RANK[rank],
        "_seq": sequence,
        "at": _iso(at),
        "label": label,
        "reference": reference,
        "montant": montant,
    }


def _sorted_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def _key(event: dict[str, Any]) -> tuple[bool, datetime, int, int]:
        stamp = event["_at"]
        missing = stamp is None
        return (
            missing,
            stamp or datetime.max.replace(tzinfo=UTC),
            int(event["_rank"]),
            int(event["_seq"]),
        )

    ordered = sorted(events, key=_key)
    for event in ordered:
        event.pop("_at", None)
        event.pop("_rank", None)
        event.pop("_seq", None)
    return ordered


def assemble_statement(
    *,
    total: Decimal | None,
    issued_at: datetime | None,
    invoice_number: str | None,
    pdf_url: str | None,
    payments: list[dict[str, Any]],
    credit_notes: list[dict[str, Any]] | None,
    refunds: list[dict[str, Any]] | None,
    balance_due: Decimal | None = None,
    refundable: Decimal | None = None,
) -> dict[str, Any]:
    """Assemble les libellés à afficher. ``None`` signifie montant inconnu."""
    documents = _documents(invoice_number, pdf_url, credit_notes)
    try:
        return _assemble(
            total=total,
            issued_at=issued_at,
            invoice_number=invoice_number,
            payments=payments,
            credit_notes=credit_notes,
            refunds=refunds,
            balance_due=balance_due,
            refundable=refundable,
            documents=documents,
        )
    except (ValueError, ArithmeticError):
        return _unavailable(_UNAVAILABLE, documents)


def _assemble(
    *,
    total: Decimal | None,
    issued_at: datetime | None,
    invoice_number: str | None,
    payments: list[dict[str, Any]],
    credit_notes: list[dict[str, Any]] | None,
    refunds: list[dict[str, Any]] | None,
    balance_due: Decimal | None,
    refundable: Decimal | None,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    sequence = 0
    if total is not None:
        events.append(
            _event(
                at=issued_at,
                rank="emission",
                label="Facture émise",
                reference=invoice_number,
                montant=_plain(total),
                sequence=sequence,
            )
        )
        sequence += 1

    deposits: list[Decimal] = []
    later: list[Decimal] = []
    for payment in payments:
        if payment.get("origine") == KIND_HANDOVER or payment.get("kind") == KIND_HANDOVER:
            continue
        amount = money(payment["amount"])
        is_deposit = payment.get("cash_collection_id") is not None
        if is_deposit:
            deposits.append(amount)
            label = "Acompte chauffeur déjà reçu"
            rank = "acompte"
        else:
            later.append(amount)
            label = "Paiement complémentaire"
            rank = "paiement"
        events.append(
            _event(
                at=payment.get("paid_at"),
                rank=rank,
                label=label,
                reference=payment.get("reference"),
                montant=_reduction(amount) if amount > 0 else _plain(amount),
                sequence=sequence,
            )
        )
        sequence += 1

    note_amounts: list[Decimal] = []
    if credit_notes is not None:
        for note in credit_notes:
            amount = money(note["amount"])
            note_amounts.append(amount)
            events.append(
                _event(
                    at=note.get("issued_at"),
                    rank="avoir",
                    label="Avoir émis",
                    reference=note.get("number"),
                    montant=_reduction(amount),
                    sequence=sequence,
                )
            )
            sequence += 1

    confirmed: list[Decimal] = []
    if refunds is not None:
        for refund in refunds:
            amount = money(refund["amount"])
            status = str(refund.get("status") or "")
            if status == "confirmed":
                confirmed.append(amount)
                label = "Remboursement confirmé"
                stamp = refund.get("confirmed_at") or refund.get("created_at")
            elif status in {"requested", "authorized"}:
                label = "Remboursement enregistré, non confirmé"
                stamp = refund.get("created_at")
            elif status in {"refused", "cancelled"}:
                label = "Remboursement refusé"
                stamp = refund.get("created_at")
            else:
                continue
            events.append(
                _event(
                    at=stamp,
                    rank="remboursement",
                    label=label,
                    reference=refund.get("reference"),
                    montant=_plain(amount),
                    sequence=sequence,
                )
            )
            sequence += 1

    payload: dict[str, Any] = {
        "disponible": True,
        "message": None,
        "events": _sorted_events(events),
        "documents": documents,
    }
    payload["acomptes_chf"] = _reduction(sum(deposits, Decimal("0.00")))
    payload["paiements_complementaires_chf"] = _reduction(sum(later, Decimal("0.00")))

    if total is None:
        payload.update(_blank_amounts())
        payload["acomptes_chf"] = _reduction(sum(deposits, Decimal("0.00")))
        payload["paiements_complementaires_chf"] = _reduction(sum(later, Decimal("0.00")))
        payload["disponible"] = balance_due is not None
        payload["message"] = None if payload["disponible"] else _UNAVAILABLE
        payload["reste_a_payer_chf"] = _plain(balance_due) if balance_due is not None else None
        payload["a_rembourser_chf"] = _plain(refundable) if refundable is not None else None
        return payload

    payload["prestations_chf"] = _plain(total)

    if credit_notes is None:
        payload["avoirs_chf"] = None
        payload["net_facture_chf"] = None
        payload["remboursements_confirmes_chf"] = None
        payload["reste_a_payer_chf"] = (
            _plain(balance_due) if balance_due is not None else None
        )
        payload["a_rembourser_chf"] = (
            _plain(refundable) if refundable is not None else None
        )
        return payload

    claim = credit_room(total, note_amounts)
    used = (money(total) - claim).quantize(Decimal("0.01"))
    payload["avoirs_chf"] = _reduction(used)
    payload["net_facture_chf"] = _plain(claim)

    if refunds is None:
        payload["remboursements_confirmes_chf"] = None
        payload["reste_a_payer_chf"] = _plain(balance_due) if balance_due is not None else None
        payload["a_rembourser_chf"] = _plain(refundable) if refundable is not None else None
        return payload

    settlement = settle(total, note_amounts, deposits + later, confirmed)
    payload["remboursements_confirmes_chf"] = _plain(sum(confirmed, Decimal("0.00")))
    payload["reste_a_payer_chf"] = _plain(settlement.balance_due_chf)
    payload["a_rembourser_chf"] = _plain(settlement.refundable_chf)
    return payload


def _table_present(qualified_name: str) -> bool:
    from sqlalchemy import text

    from ext import db

    if qualified_name == "public.invoice_credit_note":
        query = text("SELECT to_regclass('public.invoice_credit_note')")
    elif qualified_name == "public.invoice_refund":
        query = text("SELECT to_regclass('public.invoice_refund')")
    else:
        return False
    return db.session.scalar(query) is not None


def _load_credit_notes(invoice: Any) -> list[dict[str, Any]] | None:
    if not _table_present("public.invoice_credit_note"):
        return None
    from models.invoice import InvoiceCreditNote

    rows = (
        InvoiceCreditNote.query.filter_by(
            invoice_id=invoice.id, company_id=invoice.company_id
        )
        .order_by(InvoiceCreditNote.id)
        .all()
    )
    return [
        {
            "amount": row.amount_chf,
            "issued_at": row.issued_at,
            "number": row.credit_number,
            "pdf_url": row.pdf_url,
        }
        for row in rows
    ]


def _load_refunds(invoice: Any) -> list[dict[str, Any]] | None:
    if not _table_present("public.invoice_refund"):
        return None
    from models.invoice import InvoiceRefund

    rows = (
        InvoiceRefund.query.filter_by(
            invoice_id=invoice.id, company_id=invoice.company_id
        )
        .order_by(InvoiceRefund.id)
        .all()
    )
    return [
        {
            "amount": row.amount_chf,
            "status": row.status,
            "confirmed_at": row.confirmed_at,
            "created_at": row.created_at,
            "reference": str(row.id),
        }
        for row in rows
    ]


def statement_for_invoice(invoice: Any) -> dict[str, Any]:
    """Situation d'une facture patient ou clinique. N'appelle pas ``update_balance``."""
    try:
        notes = _load_credit_notes(invoice)
        refunds = _load_refunds(invoice)
        payments = [
            {
                "amount": payment.amount,
                "paid_at": payment.paid_at,
                "reference": payment.reference,
                "cash_collection_id": payment.cash_collection_id,
            }
            for payment in (invoice.payments or [])
        ]
    except (ValueError, ArithmeticError):
        return _unavailable(_UNAVAILABLE)
    result = assemble_statement(
        total=invoice.total_amount,
        issued_at=invoice.issued_at,
        invoice_number=invoice.invoice_number,
        pdf_url=invoice.pdf_url,
        payments=payments,
        credit_notes=notes,
        refunds=refunds,
        balance_due=invoice.balance_due,
        refundable=invoice.refundable_chf,
    )
    if (
        result.get("disponible")
        and notes is not None
        and refunds is not None
        and result.get("reste_a_payer_chf") is not None
        and result.get("a_rembourser_chf") is not None
    ):
        stored_balance = money(invoice.balance_due)
        stored_refundable = money(invoice.refundable_chf)
        if stored_balance != money(result["reste_a_payer_chf"]) or stored_refundable != money(
            result["a_rembourser_chf"]
        ):
            return _unavailable(_MISMATCH, result.get("documents"))
    return result
