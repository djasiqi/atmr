"""Payeur effectif d'une course patient, sans écrire en base.

Un BillingParty de type ``PATIENT`` est un destinataire technique. Il ne
constitue pas un choix explicite et ne doit pas masquer un tiers payeur
actif (curatelle, famille, etc.).

Ordre :
1. décision verrouillée ou override manuel / import
2. bon de transport valide
3. séjour actif
4. tiers déjà retenu sur la course (curatelle, famille, autre non établissement)
5. tiers payeur actif du client
6. BillingParty PATIENT technique
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ext import db
from models.billing_party import BillingParty

_EXPLICIT_SOURCES = frozenset({"manual_override", "import"})


def _type_value(party: Any) -> str:
    raw = getattr(getattr(party, "type", None), "value", None)
    if raw is None:
        raw = getattr(party, "type", None)
    return str(raw or "").lower().strip()


def _source_value(booking: Any) -> str:
    raw = getattr(getattr(booking, "billing_source", None), "value", None)
    if raw is None:
        raw = getattr(booking, "billing_source", None)
    return str(raw or "").lower().strip()


def is_technical_patient_billing_party(party: Any) -> bool:
    """Vrai pour le destinataire technique du patient, pas pour un tiers."""
    return party is not None and _type_value(party) == "patient"


def billing_choice_is_locked(booking: Any) -> bool:
    """Vrai si quelqu'un a verrouillé le payeur ou posé un override explicite."""
    if getattr(booking, "billing_locked_at", None) is not None:
        return True
    if getattr(booking, "billing_locked_by_user_id", None) is not None:
        return True
    return _source_value(booking) in _EXPLICIT_SOURCES


def _load_party(party_id: int | None) -> BillingParty | None:
    if party_id is None:
        return None
    return db.session.get(BillingParty, int(party_id))


def _party_from_resolution(payload: dict[str, Any] | None) -> BillingParty | None:
    if not payload:
        return None
    party_id = payload.get("billing_party_id")
    if party_id is None:
        return None
    return _load_party(int(party_id))


def resolve_effective_patient_billing_party(
    *,
    booking: Any,
    company_id: int,
) -> BillingParty | None:
    """Résout le payeur à facturer. Ne modifie pas le booking."""
    from services.billing.billing_party_linker import is_establishment_billing_party
    from services.billing.client_stay_resolver import (
        find_active_stay_for_booking,
        find_valid_voucher_for_booking,
        resolve_default_billing_party_for_client,
        resolve_payer_from_stay,
        resolve_payer_from_voucher,
    )

    current = _load_party(getattr(booking, "billing_party_id", None))
    if billing_choice_is_locked(booking) and current is not None:
        return current

    voucher = find_valid_voucher_for_booking(booking=booking)
    if voucher is not None:
        from_voucher = _party_from_resolution(
            resolve_payer_from_voucher(voucher=voucher, company_id=int(company_id))
        )
        if from_voucher is not None:
            return from_voucher

    stay = find_active_stay_for_booking(booking=booking)
    if stay is not None:
        from_stay = _party_from_resolution(
            resolve_payer_from_stay(stay=stay, company_id=int(company_id))
        )
        if from_stay is not None:
            return from_stay

    if (
        current is not None
        and not is_technical_patient_billing_party(current)
        and not is_establishment_billing_party(current)
    ):
        return current

    client_id = getattr(booking, "client_id", None)
    if client_id is not None:
        third = resolve_default_billing_party_for_client(
            client_id=int(client_id),
            company_id=int(company_id),
        )
        if third is not None and not is_establishment_billing_party(third):
            return third

    return current


def resolve_effective_party_id_for_bookings(
    bookings: list[Any],
    *,
    company_id: int,
) -> int | None:
    """Payeur majoritaire des courses patient. Lecture seule."""
    counts: Counter[int] = Counter()
    for booking in bookings:
        btype = str(getattr(booking, "billed_to_type", None) or "").lower().strip()
        if btype != "patient":
            continue
        party = resolve_effective_patient_billing_party(
            booking=booking,
            company_id=int(company_id),
        )
        party_id = getattr(party, "id", None)
        if party_id is None:
            party_id = getattr(booking, "billing_party_id", None)
        if party_id is not None:
            counts[int(party_id)] += 1
    if not counts:
        return None
    return min(counts, key=lambda party_id: (-counts[party_id], party_id))
