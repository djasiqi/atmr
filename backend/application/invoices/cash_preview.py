"""Lecture des acomptes chauffeur pour la prévisualisation. N'écrit rien."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import Integer, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY

from domain.end_of_trip_settlement.financial_contract import (
    CashLine,
    imputable,
    recognized_net,
)
from ext import db

_CENT = Decimal("0.01")


def _ready(table: str, column: str) -> bool:
    found = db.session.execute(
        text(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_name = :table_name AND column_name = :column_name
            """
        ),
        {"table_name": table, "column_name": column},
    ).first()
    return found is not None


def _money(value: Decimal) -> str:
    return f"{value.quantize(_CENT):.2f}"


def position_for_booking(
    contractual: Decimal,
    rows: list[dict[str, Any]],
    allocated_ids: set[int],
) -> dict[str, Any]:
    """Net reconnu d'un trajet. Le tarif contractuel n'est pas réécrit."""
    due = contractual.quantize(_CENT)
    eligible: list[int] = []
    lines: list[CashLine] = []
    for row in rows:
        if str(row.get("kind") or "collection") == "compensation":
            continue
        if row.get("net_chf") is None:
            continue
        if int(row["id"]) in allocated_ids:
            continue
        net = Decimal(row["net_chf"])
        operation_due = Decimal(row.get("amount_due_chf") or due)
        if not imputable(
            state=str(row.get("state") or ""),
            disposition=row.get("disposition"),
            net=net,
            due=operation_due,
        ):
            continue
        eligible.append(int(row["id"]))
        lines.append(
            CashLine(
                kind="collection",
                state=str(row.get("state") or ""),
                disposition=row.get("disposition"),
                net_chf=net,
                due_chf=operation_due,
            )
        )
    eligible_ids = set(eligible)
    for row in rows:
        if str(row.get("kind") or "") != "compensation":
            continue
        origin = row.get("compensates_collection_id")
        if origin is None or int(origin) not in eligible_ids:
            continue
        if row.get("net_chf") is None:
            continue
        lines.append(
            CashLine(
                kind="compensation",
                state=str(row.get("state") or "administrative"),
                disposition=row.get("disposition"),
                net_chf=Decimal(row["net_chf"]),
                due_chf=due,
            )
        )
    recognized = recognized_net(lines) if lines else Decimal("0.00")
    remaining = (due - recognized).quantize(_CENT)
    if remaining < 0:
        remaining = Decimal("0.00")
    return {
        "contractual_chf": _money(due),
        "recognized_net_chf": _money(recognized),
        "remaining_due_chf": _money(remaining),
        "imputable_collection_ids": eligible,
        "cash_imputable": recognized > 0,
    }


def merge_leg_positions(positions: list[dict[str, Any]]) -> dict[str, Any]:
    """Additionne des trajets déjà calculés. Ne répartit pas un acompte."""
    contractual = sum(
        (Decimal(item["contractual_chf"]) for item in positions), Decimal("0.00")
    )
    recognized = sum(
        (Decimal(item["recognized_net_chf"]) for item in positions), Decimal("0.00")
    )
    remaining = sum(
        (Decimal(item["remaining_due_chf"]) for item in positions), Decimal("0.00")
    )
    identifiers: list[int] = []
    for item in positions:
        identifiers.extend(int(value) for value in item["imputable_collection_ids"])
    return {
        "contractual_chf": _money(contractual),
        "recognized_net_chf": _money(recognized),
        "remaining_due_chf": _money(remaining),
        "imputable_collection_ids": identifiers,
        "cash_imputable": recognized > 0,
        "cash_legs": list(positions),
    }


def load_patient_cash_preview(
    company_id: int, dues: dict[int, Decimal]
) -> dict[int, dict[str, Any]] | None:
    """Positions par réservation. None si le schéma caisse multiligne n'est pas là."""
    if not dues:
        return {}
    if not _ready("cash_collection", "kind") or not _ready(
        "invoice_payments", "cash_collection_id"
    ):
        return None
    booking_ids = [int(booking_id) for booking_id in dues]
    rows = (
        db.session.execute(
            text(
                """
                SELECT id, booking_id, company_id, kind, state, disposition,
                       net_chf, amount_due_chf, compensates_collection_id
                FROM cash_collection
                WHERE company_id = :company_id
                  AND booking_id = ANY(:booking_ids)
                ORDER BY id
                """
            ).bindparams(bindparam("booking_ids", type_=ARRAY(Integer))),
            {"company_id": int(company_id), "booking_ids": booking_ids},
        )
        .mappings()
        .all()
    )
    collection_ids = [int(row["id"]) for row in rows]
    allocated: set[int] = set()
    if collection_ids:
        linked = db.session.execute(
            text(
                """
                SELECT cash_collection_id
                FROM invoice_payments
                WHERE cash_collection_id = ANY(:collection_ids)
                """
            ).bindparams(bindparam("collection_ids", type_=ARRAY(Integer))),
            {"collection_ids": collection_ids},
        ).scalars()
        allocated = {int(value) for value in linked if value is not None}
    grouped: dict[int, list[dict[str, Any]]] = {
        booking_id: [] for booking_id in booking_ids
    }
    for row in rows:
        if int(row["company_id"]) != int(company_id):
            continue
        grouped[int(row["booking_id"])].append(dict(row))
    return {
        booking_id: position_for_booking(
            Decimal(dues[booking_id]), grouped[booking_id], allocated
        )
        | {"booking_id": booking_id}
        for booking_id in booking_ids
    }
