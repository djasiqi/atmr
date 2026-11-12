"""Contrat de lecture chauffeur. La même fonction sert le GET et le garde d'écriture.

Le garde d'écriture ne recopie pas les refus de `refuse_booking`. Il demande
seulement si cette lecture proposerait les espèces.
"""

from __future__ import annotations

from typing import Any

from domain.end_of_trip_settlement.recipe_simulation import (
    RecipeBookingFacts,
    overlay_recipe_contract,
    recipe_simulation_read_allowed,
    simulation_policy,
)


def _safe_contract(contract: dict[str, object]) -> dict[str, object]:
    safe = dict(contract)
    safe["payment_executable"] = False
    safe["persists_payment"] = False
    safe["writes_completed_payment"] = False
    safe["writes_cash_receipt"] = False
    safe["writes_invoice"] = False
    safe["writes_financial_signature"] = False
    if not safe.get("simulation"):
        safe["modes"] = []
        safe["preview_screen"] = None
    return safe


def recipe_overlay(booking_id: int, contract: dict[str, object]) -> dict[str, object]:
    """N'enveloppe qu'une revue administrative patient portant une politique simulée."""
    from ext import db
    from models.booking import Booking
    from models.settlement_policy import BookingSettlementPolicy

    booking = db.session.get(Booking, booking_id)
    if booking is None:
        return contract
    root_id = int(booking.parent_booking_id or booking.id)
    siblings = (
        db.session.query(Booking.id)
        .filter((Booking.id == root_id) | (Booking.parent_booking_id == root_id))
        .count()
    )
    policy = (
        db.session.query(BookingSettlementPolicy)
        .filter_by(booking_id=booking_id, status="active")
        .one_or_none()
    )
    simulation_modes = None
    simulation_modality = None
    if policy is not None and simulation_policy(policy.policy_version):
        simulation_modes = tuple(str(mode) for mode in (policy.authorized_modes or []))
        simulation_modality = str(policy.modality)
    from application.settlement.production_gates import contractual_price_label

    amount_label = contractual_price_label(booking.price_amount)
    from application.settlement.leg_settlement import separate_leg_ready

    round_trip = siblings > 1 or bool(booking.is_return) or bool(booking.is_round_trip)
    facts = RecipeBookingFacts(
        billed_to_type=str(
            getattr(booking.billed_to_type, "value", booking.billed_to_type) or ""
        ),
        is_round_trip=round_trip,
        amount_chf=amount_label,
        customer_name=getattr(booking, "customer_name", None),
        company_id=int(booking.company_id),
        simulation_modes=simulation_modes,
        simulation_modality=simulation_modality,
        separate_leg_approved=bool(round_trip and separate_leg_ready(booking)),
    )
    return overlay_recipe_contract(contract, facts)


def driver_settlement_contract(
    booking_id: int, driver: Any
) -> dict[str, object] | None:
    """Même assemblage que GET settlement-read. N'écrit rien."""
    from application.drivers.read_mission_settlement import load_mission_decision

    contract = load_mission_decision(int(booking_id), driver)
    if contract is None:
        return None
    if recipe_simulation_read_allowed():
        contract = recipe_overlay(int(booking_id), contract)
    safe = _safe_contract(contract)
    if recipe_simulation_read_allowed():
        from application.settlement.patient_billing_prep import billing_followup_hint

        hint = billing_followup_hint(int(booking_id))
        if hint is not None:
            safe["billing_followup"] = hint
        from application.settlement.leg_settlement import exceptional_cash_hint

        offer = exceptional_cash_hint(int(booking_id))
        if offer is not None:
            safe["exceptional_cash"] = offer
    return _publish_contractual_read(int(booking_id), driver, safe)


def _publish_contractual_read(
    booking_id: int, driver: Any, contract: dict[str, object]
) -> dict[str, object]:
    """Expose le prix confirmé et le motif de la feuille, sans l'ouvrir.

    ``amount`` ne remplace pas ``price_amount``. Une politique de simulation
    absente ne devient pas une autorisation d'espèces.
    """
    from application.settlement.production_gates import (
        contractual_price_label,
        debtor_label,
    )
    from domain.end_of_trip_settlement.production_prep_barrier import prep_open
    from ext import db
    from models.booking import Booking
    from models.company import Company

    booking = db.session.get(Booking, booking_id)
    if booking is None:
        return contract
    published = dict(contract)
    label = contractual_price_label(booking.price_amount)
    if published.get("amount_chf") in (None, "") and label is not None:
        published["amount_chf"] = label
    if not published.get("billed_to_type"):
        raw_payer = getattr(booking.billed_to_type, "value", booking.billed_to_type)
        if raw_payer:
            published["billed_to_type"] = str(raw_payer)
    if not published.get("debtor_label"):
        published["debtor_label"] = debtor_label(booking)
    company = db.session.get(Company, int(booking.company_id))
    company_name = getattr(company, "name", None) if company is not None else None
    if company_name and not published.get("company_name"):
        published["company_name"] = str(company_name)
    published["cash_sheet"] = _cash_sheet(booking, driver, published, prep_open())
    return published


def _cash_sheet(
    booking: Any, driver: Any, contract: dict[str, object], prep: bool
) -> dict[str, object]:
    from application.settlement.production_gates import refuse_booking
    from domain.end_of_trip_settlement.production_prep_barrier import PrepRefusal

    offered = cash_mode_offered(contract)
    if offered:
        return {
            "allowed": True,
            "reason": None,
            "real_collection": False,
            "simulation": True,
            "register": "simulation",
        }
    reason = "especes_non_autorisees"
    if not prep:
        reason = "prep_closed"
    else:
        reader_id = getattr(driver, "id", None)
        reader_company = getattr(driver, "company_id", None)
        if reader_id is not None and reader_company is not None:
            try:
                refuse_booking(
                    booking,
                    driver_id=int(reader_id),
                    company_id=int(reader_company),
                )
            except PrepRefusal as refusal:
                reason = str(refusal.code)
    from domain.end_of_trip_settlement.v1_local_ledger import v1_local_ledger_enabled

    if reason == "especes_non_autorisees" and prep and v1_local_ledger_enabled():
        return {
            "allowed": True,
            "reason": None,
            "real_collection": False,
            "simulation": False,
            "register": "cash_v1",
            "local_ledger": True,
        }
    if reason == "especes_non_autorisees" and prep:
        # Prep ouverte sur base jetable : feuille de caisse visible, sans activation produit.
        return {
            "allowed": True,
            "reason": None,
            "real_collection": False,
            "simulation": False,
            "register": "cash_prep",
            "production_prep": True,
        }
    return {
        "allowed": False,
        "reason": reason,
        "real_collection": False,
        "simulation": False,
        "register": "cash_v1",
    }


def cash_mode_offered(contract: dict[str, object] | None) -> bool:
    """Vrai seulement si la lecture propose des espèces simulées."""
    if not contract:
        return False
    if contract.get("simulation") is not True or contract.get("source") != "simulation":
        return False
    if contract.get("stale") is True:
        return False
    if contract.get("kind") != "collection_authorized":
        return False
    modes = contract.get("modes")
    if isinstance(modes, list | tuple):
        return "cash" in modes
    return False
