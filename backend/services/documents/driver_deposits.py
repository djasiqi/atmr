"""Acomptes chauffeur déjà imputés, lus uniquement sur les paiements.

Le montant affiché est celui **imputé** à la facture (`invoice_payments.amount`),
jamais le brut éventuellement plus élevé de `cash_collection`.
Aucune collection non liée n'est relue. Aucune écriture n'est faite.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import Integer, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY

_CENT = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class DriverDepositLine:
    """Une imputation d'acompte traçable pour le PDF."""

    payment_id: int
    amount_chf: Decimal
    paid_at: datetime | None
    method: str
    cash_collection_id: int | None
    booking_id: int | None
    booking_scheduled_at: datetime | None
    mode_label: str


def _method_value(payment: Any) -> str:
    method = getattr(payment, "method", None)
    return str(getattr(method, "value", method) or "")


def _mode_label(method: str, *, has_collection: bool) -> str:
    normalized = (method or "").strip().lower()
    if normalized == "cash" and has_collection:
        return "Espèces chauffeur"
    if normalized == "cash":
        return "Espèces"
    if normalized in {"card", "credit_card"}:
        return "Carte"
    if normalized in {"bank_transfer", "transfer"}:
        return "Virement"
    if normalized:
        return normalized.replace("_", " ").capitalize()
    return "Paiement"


def _collection_booking_map(collection_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not collection_ids:
        return {}
    try:
        from ext import db

        rows = (
            db.session.execute(
                text(
                    """
                    SELECT c.id AS collection_id,
                           c.booking_id,
                           b.scheduled_time
                    FROM cash_collection AS c
                    LEFT JOIN booking AS b ON b.id = c.booking_id
                    WHERE c.id = ANY(:collection_ids)
                    """
                ).bindparams(bindparam("collection_ids", type_=ARRAY(Integer))),
                {"collection_ids": collection_ids},
            )
            .mappings()
            .all()
        )
    except Exception:
        # Hors contexte Flask / schéma absent : montant imputé reste affiché.
        return {}
    return {
        int(row["collection_id"]): {
            "booking_id": int(row["booking_id"])
            if row["booking_id"] is not None
            else None,
            "scheduled_time": row.get("scheduled_time"),
        }
        for row in rows
    }


def persisted_driver_deposit_lines(invoice: Any) -> list[DriverDepositLine]:
    """Lignes d'imputation cash liées à une collection, sans doublon de paiement."""
    payments = getattr(invoice, "payments", None) or []
    cash_payments: list[Any] = []
    collection_ids: list[int] = []
    for payment in payments:
        if _method_value(payment) != "cash":
            continue
        collection_id = getattr(payment, "cash_collection_id", None)
        if collection_id is None:
            continue
        try:
            amount = Decimal(str(payment.amount)).quantize(_CENT)
        except (ArithmeticError, ValueError, TypeError):
            continue
        if amount <= 0:
            continue
        cash_payments.append(payment)
        collection_ids.append(int(collection_id))
    booking_by_collection = _collection_booking_map(collection_ids)

    lines: list[DriverDepositLine] = []
    seen_payment_ids: set[int] = set()
    for payment in cash_payments:
        payment_id = getattr(payment, "id", None)
        try:
            pid = int(payment_id) if payment_id is not None else 0
        except (TypeError, ValueError):
            pid = 0
        if pid and pid in seen_payment_ids:
            continue
        if pid:
            seen_payment_ids.add(pid)
        collection_id = int(payment.cash_collection_id)
        meta = booking_by_collection.get(collection_id) or {}
        scheduled = meta.get("scheduled_time")
        method = _method_value(payment)
        lines.append(
            DriverDepositLine(
                payment_id=pid,
                amount_chf=Decimal(str(payment.amount)).quantize(_CENT),
                paid_at=getattr(payment, "paid_at", None),
                method=method,
                cash_collection_id=collection_id,
                booking_id=meta.get("booking_id"),
                booking_scheduled_at=scheduled
                if isinstance(scheduled, datetime)
                else None,
                mode_label=_mode_label(method, has_collection=True),
            )
        )
    return lines


def persisted_driver_deposits(invoice: Any) -> list[Decimal]:
    """Montants cash déjà liés à une collection, dans l'ordre des paiements."""
    return [line.amount_chf for line in persisted_driver_deposit_lines(invoice)]


def driver_deposit_total(invoice: Any) -> Decimal:
    return sum(persisted_driver_deposits(invoice), Decimal("0.00")).quantize(_CENT)


def _amount_paid(invoice: Any) -> Decimal:
    try:
        return Decimal(str(getattr(invoice, "amount_paid", 0) or 0)).quantize(_CENT)
    except (ArithmeticError, ValueError, TypeError):
        return Decimal("0.00")


def _is_declared_cash(line: DriverDepositLine) -> bool:
    """Espèces sans collecte : règlement manuel ou montant seulement déclaré."""
    return (
        line.method or ""
    ).strip().lower() == "cash" and line.cash_collection_id is None


def _lines_covered_by_amount_paid(
    lines: list[DriverDepositLine], amount_paid: Decimal
) -> list[DriverDepositLine]:
    """Ne présente comme encaissé que ce que ``amount_paid`` couvre.

    Les écritures confirmées (virement, carte, espèces liées à une collecte)
    sont prises en premier. Un montant espèces sans collecte n'est gardé que
    s'il tient encore dans le reste encaissé : c'est alors un règlement manuel.
    Le surplus est un montant déclaré, non présenté comme encaissé.
    """
    remaining = amount_paid if amount_paid > 0 else Decimal("0.00")
    confirmed = [line for line in lines if not _is_declared_cash(line)]
    declared_cash = [line for line in lines if _is_declared_cash(line)]
    covered: list[DriverDepositLine] = []
    for line in confirmed + declared_cash:
        if line.amount_chf <= remaining:
            remaining -= line.amount_chf
            covered.append(line)
    covered_ids = {id(line) for line in covered}
    return [line for line in lines if id(line) in covered_ids]


def persisted_invoice_payment_lines(invoice: Any) -> list[DriverDepositLine]:
    """Paiements positifs réellement couverts par ``amount_paid``.

    Inclut virement, carte et espèces manuelles imputées. Les espèces liées
    à une collecte portent le ``booking_id`` lorsque disponible. Un montant
    espèces sans collecte qui dépasse le montant encaissé n'est pas une
    écriture : il n'apparaît pas.
    """
    cash_linked = {
        int(line.payment_id): line
        for line in persisted_driver_deposit_lines(invoice)
        if line.payment_id
    }
    payments = getattr(invoice, "payments", None) or []
    lines: list[DriverDepositLine] = []
    seen: set[int] = set()
    for payment in payments:
        payment_id = getattr(payment, "id", None)
        try:
            pid = int(payment_id) if payment_id is not None else 0
        except (TypeError, ValueError):
            pid = 0
        if pid and pid in seen:
            continue
        try:
            amount = Decimal(str(payment.amount)).quantize(_CENT)
        except (ArithmeticError, ValueError, TypeError):
            continue
        if amount <= 0:
            continue
        if pid:
            seen.add(pid)
        if pid and pid in cash_linked:
            lines.append(cash_linked[pid])
            continue
        method = _method_value(payment)
        collection_id = getattr(payment, "cash_collection_id", None)
        try:
            cid = int(collection_id) if collection_id is not None else None
        except (TypeError, ValueError):
            cid = None
        lines.append(
            DriverDepositLine(
                payment_id=pid,
                amount_chf=amount,
                paid_at=getattr(payment, "paid_at", None),
                method=method,
                cash_collection_id=cid,
                booking_id=None,
                booking_scheduled_at=None,
                mode_label=_mode_label(method, has_collection=cid is not None),
            )
        )
    return _lines_covered_by_amount_paid(lines, _amount_paid(invoice))
