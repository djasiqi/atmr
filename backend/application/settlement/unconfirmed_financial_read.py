"""Contrat de lecture quand aucune politique de règlement n'est consultable.

La réservation connue ne suffit pas à décider un règlement chauffeur.
Aucune ligne n'est créée, aucun moyen n'est autorisé, aucun paiement n'est écrit.
"""

from __future__ import annotations

from typing import Any


def _amount(value: Any) -> str | None:
    from application.settlement.production_gates import contractual_price_label

    return contractual_price_label(value)


def unconfirmed_financial_contract(
    *, billed_to_type: str | None, amount: Any
) -> dict[str, Any]:
    payer = (billed_to_type or "").strip().lower() or None
    return {
        "kind": "financial_situation_unconfirmed",
        "source": "booking_only",
        "verification_required": True,
        "reason": "financial_policy_unavailable",
        "billed_to_type": payer,
        "amount_chf": _amount(amount),
        "modes": [],
        "simulation": False,
        "payment_executable": False,
        "persists_payment": False,
        "authorization_durable": False,
        "writes_completed_payment": False,
        "writes_cash_receipt": False,
        "writes_invoice": False,
        "writes_financial_signature": False,
        "driver_may_edit_amount": False,
        "driver_may_edit_payer": False,
        "stale": False,
    }


def read_unconfirmed_financial_contract(
    booking_id: int, driver: Any
) -> dict[str, Any] | None:
    from ext import db
    from models.booking import Booking

    booking = db.session.get(Booking, int(booking_id))
    driver_id = getattr(driver, "id", None)
    if booking is None or driver_id is None or booking.driver_id != driver_id:
        return None
    amount = booking.price_amount
    billed = booking.billed_to_type if isinstance(booking.billed_to_type, str) else None
    return unconfirmed_financial_contract(billed_to_type=billed, amount=amount)
