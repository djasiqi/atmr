"""Règlement d'un trajet d'aller-retour. Le garde-fou de groupe reste le défaut.

Un trajet n'est autorisé que si l'entreprise a approuvé le règlement séparé,
que le montant est celui de ce trajet, et qu'un trajet lié existe.
Le prix total n'est jamais coupé en deux.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from domain.end_of_trip_settlement.production_prep_barrier import PrepRefusal
from ext import db

_CENT = Decimal("0.01")


def linked_round_trip(booking: Any) -> bool:
    return bool(
        getattr(booking, "is_round_trip", False)
        or getattr(booking, "is_return", False)
        or getattr(booking, "parent_booking_id", None)
    )


def sibling_of(booking: Any) -> Any | None:
    from models.booking import Booking

    parent_id = getattr(booking, "parent_booking_id", None)
    if parent_id is not None:
        return db.session.get(Booking, int(parent_id))
    return (
        db.session.query(Booking)
        .filter(Booking.parent_booking_id == int(booking.id))
        .order_by(Booking.id)
        .first()
    )


def leg_role(booking: Any) -> str:
    if getattr(booking, "parent_booking_id", None) or getattr(booking, "is_return", False):
        return "return"
    if getattr(booking, "is_round_trip", False) or sibling_of(booking) is not None:
        return "outbound"
    return "single"


def _money(raw: Any) -> Decimal | None:
    if raw is None:
        return None
    try:
        amount = Decimal(str(raw)).quantize(_CENT)
    except Exception:
        return None
    if amount <= 0:
        return None
    return amount


def _active_policy(booking_id: int) -> Any | None:
    from models.settlement_policy import BookingSettlementPolicy

    return (
        db.session.query(BookingSettlementPolicy)
        .filter_by(booking_id=int(booking_id), status="active")
        .one_or_none()
    )


def separate_leg_ready(booking: Any) -> bool:
    """Vrai seulement pour ce trajet. Un aller-retour sans accord reste bloqué."""
    if not linked_round_trip(booking):
        return False
    if getattr(booking, "id", None) is None:
        return False
    sibling = sibling_of(booking)
    if sibling is None or int(sibling.company_id) != int(booking.company_id):
        return False
    if _money(getattr(booking, "price_amount", None)) is None:
        return False
    if _money(getattr(sibling, "price_amount", None)) is None:
        return False
    payer = str(
        getattr(
            getattr(booking, "billed_to_type", None),
            "value",
            getattr(booking, "billed_to_type", None),
        )
        or ""
    ).strip().lower()
    if payer != "patient":
        return False
    policy = _active_policy(int(booking.id))
    if policy is None:
        return False
    if not bool(policy.separate_leg_collection_approved):
        return False
    if bool(policy.round_trip_collection_approved):
        return False
    if str(policy.payment_scope or "") != "booking":
        return False
    due = _money(getattr(booking, "price_amount", None))
    imposed = _money(getattr(policy, "amount_chf", None))
    return imposed is not None and due is not None and imposed == due


def leg_public(booking: Any) -> dict[str, Any]:
    """Lecture. Le montant annoncé est celui du trajet, jamais la somme."""
    role = leg_role(booking)
    due = _money(getattr(booking, "price_amount", None))
    sibling = sibling_of(booking) if role != "single" else None
    sibling_due = None if sibling is None else _money(getattr(sibling, "price_amount", None))
    return {
        "leg_role": role,
        "leg_amount_chf": None if due is None else f"{due:.2f}",
        "linked_booking_id": None if sibling is None else int(sibling.id),
        "sibling_amount_chf": None if sibling_due is None else f"{sibling_due:.2f}",
        "separate_leg_authorized": separate_leg_ready(booking) if role != "single" else False,
        "amounts_added": False,
    }


def reject_leg_amount(booking: Any, payload: dict, *, company_id: int) -> None:
    """Refuse la somme des deux trajets et une moitié calculée."""
    from application.settlement.production_gates import (
        amount_due,
        reject_client_overrides,
    )

    due = amount_due(booking)
    sibling = sibling_of(booking) if linked_round_trip(booking) else None
    sibling_due = None if sibling is None else _money(getattr(sibling, "price_amount", None))
    raw = payload.get("amount_chf")
    if raw is not None and sibling_due is not None:
        try:
            offered = Decimal(str(raw)).quantize(_CENT)
        except Exception as exc:
            raise PrepRefusal("montant_client_ignore", 400) from exc
        total = (due + sibling_due).quantize(_CENT)
        half = (total / Decimal(2)).quantize(_CENT)
        if offered == total and offered != due:
            raise PrepRefusal("montant_groupe_refuse", 400)
        if offered == half and offered != due:
            raise PrepRefusal("repartition_moitie_refusee", 400)
    reject_client_overrides(payload, due=due, company_id=company_id)


def exceptional_cash_allowed(booking: Any) -> bool:
    """Espèces pendant une mission destinée à la facture, mandat applicable seulement."""
    if getattr(booking, "invoice_line_id", None) is not None:
        return False
    if linked_round_trip(booking) and not separate_leg_ready(booking):
        return False
    payer = str(
        getattr(
            getattr(booking, "billed_to_type", None),
            "value",
            getattr(booking, "billed_to_type", None),
        )
        or ""
    ).strip().lower()
    if payer != "patient":
        return False
    if getattr(booking, "id", None) is None or getattr(booking, "client_id", None) is None:
        return False
    policy = _active_policy(int(booking.id))
    if policy is None:
        return False
    from domain.end_of_trip_settlement.recipe_simulation import simulation_policy

    if not simulation_policy(getattr(policy, "policy_version", None)):
        return False
    if str(policy.modality or "") != "patient_invoice_after":
        return False
    modes = tuple(str(mode) for mode in (policy.authorized_modes or []))
    if "cash" in modes:
        return False
    from application.settlement.patient_billing_prep import _situation

    kind = str(_situation(booking).get("kind") or "absent")
    return kind == "applicable_demonstration"


def require_exceptional_cash(booking: Any) -> None:
    if not exceptional_cash_allowed(booking):
        raise PrepRefusal("encaissement_exceptionnel_refuse", 409)


def exceptional_cash_hint(booking_id: int) -> dict[str, Any] | None:
    """Lecture seule. N'ouvre pas la feuille espèces du tableau de bord."""
    from domain.end_of_trip_settlement.recipe_simulation import (
        recipe_simulation_read_allowed,
    )
    from models.booking import Booking

    if not recipe_simulation_read_allowed():
        return None
    booking = db.session.get(Booking, int(booking_id))
    if booking is None or not exceptional_cash_allowed(booking):
        return None
    due = _money(getattr(booking, "price_amount", None))
    if due is None:
        return None
    return {
        "allowed": True,
        "amount_chf": f"{due:.2f}",
        "label": "Encaisser en espèces",
        "changes_payer": False,
        "changes_tariff": False,
        "changes_mandate": False,
        "invoice_issued": False,
        "real_payment": False,
    }
