"""Imputation des acomptes chauffeur au moment de l'émission.

La facture, ses lignes et les paiements partagent la transaction de l'appelant.
Ce module n'ouvre pas de seconde génération et ne produit pas de PDF.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import Integer, bindparam, text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import IntegrityError

from domain.end_of_trip_settlement.financial_contract import imputable, settle
from ext import db

_CENT = Decimal("0.01")


class AllocationRefusal(Exception):
    """Refus d'imputation. L'appelant annule toute la transaction."""

    def __init__(self, code: str, status: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def patient_cash_applies(
    *,
    clinic_company_id: int | None,
    bill_to_client_id: int | None,
    institution_patient_id: int | None,
) -> bool:
    """Les parcours clinique, institutionnel et partenaire ne lisent pas la caisse."""
    return (
        clinic_company_id is None
        and bill_to_client_id is None
        and institution_patient_id is None
    )


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


def _cash_schema_ready() -> bool:
    return _ready("cash_collection", "kind") and _ready(
        "invoice_payments", "cash_collection_id"
    )


def _lock_bookings(booking_ids: list[int]) -> list[dict[str, Any]]:
    if not booking_ids:
        return []
    rows = (
        db.session.execute(
            text(
                """
                SELECT id, company_id, invoice_line_id, price_amount
                FROM booking
                WHERE id = ANY(:booking_ids)
                ORDER BY id
                FOR UPDATE
                """
            ).bindparams(bindparam("booking_ids", type_=ARRAY(Integer))),
            {"booking_ids": booking_ids},
        )
        .mappings()
        .all()
    )
    return [dict(row) for row in rows]


def find_patient_cash_replay(company_id: int, booking_ids: list[int]) -> int | None:
    """Retrouve une émission déjà liée, sans créer de paiement.

    None si au moins une réservation demandée n'est pas encore facturée.
    """
    ordered = sorted({int(booking_id) for booking_id in booking_ids})
    if not ordered:
        return None
    locked = _lock_bookings(ordered)
    if len(locked) != len(ordered):
        return None
    if any(int(row["company_id"] or 0) != int(company_id) for row in locked):
        raise AllocationRefusal("autre_entreprise")
    line_ids = [int(row["invoice_line_id"]) for row in locked if row["invoice_line_id"]]
    if len(line_ids) != len(locked):
        return None
    unique_lines = list(dict.fromkeys(line_ids))
    invoices = (
        db.session.execute(
            text(
                """
                SELECT line.id AS line_id, line.invoice_id, invoice.company_id
                FROM invoice_lines AS line
                JOIN invoices AS invoice ON invoice.id = line.invoice_id
                WHERE line.id = ANY(:line_ids)
                """
            ).bindparams(bindparam("line_ids", type_=ARRAY(Integer))),
            {"line_ids": unique_lines},
        )
        .mappings()
        .all()
    )
    invoice_ids = {int(row["invoice_id"]) for row in invoices}
    if len(invoices) != len(unique_lines) or len(invoice_ids) != 1:
        raise AllocationRefusal("ambigu")
    invoice_company = int(invoices[0]["company_id"])
    if invoice_company != int(company_id):
        raise AllocationRefusal("autre_entreprise")
    return next(iter(invoice_ids))


def _shares(
    rows: list[dict[str, Any]], allocated: set[int], *, company_id: int
) -> list[tuple[int, Decimal]]:
    eligible: list[dict[str, Any]] = []
    for row in rows:
        if int(row["company_id"]) != int(company_id):
            if str(row.get("kind") or "collection") == "collection":
                raise AllocationRefusal("autre_entreprise")
            continue
        if str(row.get("kind") or "collection") == "compensation":
            continue
        if row.get("net_chf") is None:
            continue
        net = Decimal(row["net_chf"])
        due = Decimal(row.get("amount_due_chf") or "0")
        counts = imputable(
            state=str(row.get("state") or ""),
            disposition=row.get("disposition"),
            net=net,
            due=due,
        )
        if int(row["id"]) in allocated:
            if counts:
                raise AllocationRefusal("collection_deja_imputee")
            continue
        if not counts:
            continue
        eligible.append(row)
    shares: list[tuple[int, Decimal]] = []
    eligible_ids = {int(row["id"]) for row in eligible}
    for row in eligible:
        net = Decimal(row["net_chf"])
        for compensation in rows:
            if str(compensation.get("kind") or "") != "compensation":
                continue
            origin = compensation.get("compensates_collection_id")
            if origin is None or int(origin) != int(row["id"]):
                continue
            if int(origin) not in eligible_ids:
                continue
            if compensation.get("net_chf") is None:
                continue
            net += Decimal(compensation["net_chf"])
        net = net.quantize(_CENT)
        if net < 0:
            raise AllocationRefusal("depassement_autorise")
        if net == 0:
            continue
        shares.append((int(row["id"]), net))
    return shares


def impute_confirmed_cash(
    company_id: int,
    invoice_id: int,
    total: Decimal,
    dues: dict[int, Decimal],
) -> tuple[Decimal, Decimal]:
    """Crée les paiements d'acompte. Le tarif des lignes n'est pas modifié.

    Retourne le payé et le solde. Si le schéma caisse est absent, le solde
    reste le total et aucun paiement n'est inventé.
    """
    if not dues or not _cash_schema_ready():
        settlement = settle(total, [], [], [])
        return settlement.net_paid_chf, settlement.balance_due_chf
    booking_ids = sorted(int(booking_id) for booking_id in dues)
    _lock_bookings(booking_ids)
    rows = (
        db.session.execute(
            text(
                """
                SELECT id, booking_id, company_id, kind, state, disposition,
                       net_chf, amount_due_chf, compensates_collection_id
                FROM cash_collection
                WHERE booking_id = ANY(:booking_ids)
                ORDER BY id
                FOR UPDATE
                """
            ).bindparams(bindparam("booking_ids", type_=ARRAY(Integer))),
            {"booking_ids": booking_ids},
        )
        .mappings()
        .all()
    )
    loaded = [dict(row) for row in rows]
    collection_ids = [int(row["id"]) for row in loaded]
    allocated: set[int] = set()
    if collection_ids:
        linked = db.session.execute(
            text(
                """
                SELECT cash_collection_id
                FROM invoice_payments
                WHERE cash_collection_id = ANY(:collection_ids)
                FOR UPDATE
                """
            ).bindparams(bindparam("collection_ids", type_=ARRAY(Integer))),
            {"collection_ids": collection_ids},
        ).scalars()
        allocated = {int(value) for value in linked if value is not None}
    shares = _shares(loaded, allocated, company_id=int(company_id))
    paid_amounts = [amount for _collection_id, amount in shares]
    settlement = settle(total, [], paid_amounts, [])
    if settlement.balance_due_chf < 0:
        raise AllocationRefusal("depassement_autorise")
    for collection_id, amount in shares:
        try:
            db.session.execute(
                text(
                    """
                    INSERT INTO invoice_payments
                        (invoice_id, amount, method, cash_collection_id, paid_at)
                    VALUES
                        (:invoice_id, :amount, 'cash', :collection_id, now())
                    """
                ),
                {
                    "invoice_id": int(invoice_id),
                    "amount": amount,
                    "collection_id": int(collection_id),
                },
            )
        except IntegrityError as exc:
            raise AllocationRefusal("collection_deja_imputee") from exc
    db.session.execute(
        text(
            """
            UPDATE invoices
            SET amount_paid = :amount_paid, balance_due = :balance_due
            WHERE id = :invoice_id AND company_id = :company_id
            """
        ),
        {
            "amount_paid": settlement.net_paid_chf,
            "balance_due": settlement.balance_due_chf,
            "invoice_id": int(invoice_id),
            "company_id": int(company_id),
        },
    )
    return settlement.net_paid_chf, settlement.balance_due_chf
