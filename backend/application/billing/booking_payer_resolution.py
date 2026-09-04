"""Résolution atomique du triplet payeur booking (institution)."""

from __future__ import annotations

from typing import Any

from models import Booking, TransportRequest
from services.billing.institution_billing_resolver import (
    STATUS_SUCCESS,
    resolve_billing_party_for_institution_booking,
)


def billing_intent_for_billed_to_type(billed_to_type: str) -> str:
    t = (billed_to_type or "patient").lower().strip()
    if t == "clinic":
        return "institution"
    if t == "insurance":
        return "insurance"
    return "patient"


def normalize_institution_target_payer(raw: str) -> str:
    t = (raw or "patient").lower().strip()
    if t in ("institution", "clinic"):
        return "clinic"
    if t == "patient":
        return "patient"
    msg = f"Payeur cible invalide : {raw!r} (patient ou clinic attendu)."
    raise ValueError(msg)


def apply_institution_payer_resolution(
    booking: Booking,
    *,
    target_billed_to_type: str,
    transport_request: TransportRequest | None = None,
) -> dict[str, Any]:
    """Applique le resolver institutionnel — mutates ``booking`` in-place."""
    company_id = getattr(booking, "company_id", None)
    if company_id is None:
        msg = "Booking sans company_id : résolution payeur impossible."
        raise ValueError(msg)

    if transport_request is None:
        resolve_fn = getattr(booking, "_resolve_source_transport_request", None)
        transport_request = resolve_fn() if callable(resolve_fn) else None
    if transport_request is None:
        msg = "TransportRequest institution introuvable pour ce booking."
        raise ValueError(msg)

    target = normalize_institution_target_payer(target_billed_to_type)
    billing_intent = billing_intent_for_billed_to_type(target)
    if getattr(transport_request, "billing_intent", None) != billing_intent:
        transport_request.billing_intent = billing_intent

    result = resolve_billing_party_for_institution_booking(
        booking=booking,
        transport_request=transport_request,
        company_id=int(company_id),
        billing_intent_override=billing_intent,
    )

    from services.billing.billing_party_linker import (
        ensure_patient_destination_billing_party,
    )

    if target == "patient":
        booking.billed_to_type = "patient"
        ensure_patient_destination_billing_party(booking)
    elif target == "clinic":
        booking.billed_to_type = "clinic"

    status = str(result.get("billing_resolution_status") or "")
    if status != STATUS_SUCCESS:
        msg = (
            f"Résolution payeur en échec ({status}) "
            f"pour billing_intent={billing_intent}."
        )
        raise ValueError(msg)

    if target == "clinic" and not getattr(booking, "billing_party_id", None):
        raise ValueError("Résolution clinique sans billing_party_id.")

    if target == "patient":
        if getattr(booking, "billed_to_company_id", None) is not None:
            booking.billed_to_company_id = None
        if not getattr(booking, "billing_party_id", None):
            raise ValueError("Résolution patient sans billing_party_id.")

    return result
