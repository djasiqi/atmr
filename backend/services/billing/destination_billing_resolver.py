"""Résolution facturation par destination (multi-payeurs sur une mission).

Règle métier LIRIE (autoritaire) — payeur effectif d'un segment :

    effective_billing_intent =
        destination_billing_override   si défini (use_custom_billing=true)
        sinon billing_intent           (payeur principal « Facturé à »)

Chaque segment (aller principal, étape intermédiaire, retour) est évalué
**indépendamment**. Un override Patient sur une étape ne se propage pas à
l'étape suivante ni au retour : celles-ci héritent du payeur principal
tant qu'elles n'ont pas leur propre mention spécifique.

Le booking parent n'est **pas** une source de vérité pour la facturation
d'un retour ou d'une étape enfant.

Sources persistées :
- ``TransportRequest.billing_intent`` — payeur principal
- ``TransportRequestLeg.destination_billing_override`` — exception par segment
  (NULL = pas d'exception, héritage du principal)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from models.transport_request import TransportRequest
    from models.transport_request_leg import TransportRequestLeg

_BILLING_INTENT_LABELS: dict[str, str] = {
    "patient": "Patient",
    "institution": "Institution",
    "curator": "Curateur",
    "spc": "SPC",
    "insurance": "Assurance",
    "other": "Autre",
}


def billing_intent_label(intent: str | None) -> str:
    if not intent:
        return "Patient"
    return _BILLING_INTENT_LABELS.get(str(intent).lower(), str(intent))


def resolve_effective_billing_intent(
    primary: str | None,
    destination_override: str | None,
) -> str:
    """Payeur effectif d'une destination : override destination ou principal."""
    if destination_override:
        return str(destination_override).lower()
    return (primary or "patient").lower()


def effective_billing_for_leg(
    leg: TransportRequestLeg,
    transport_request: TransportRequest,
) -> str:
    override = getattr(leg, "destination_billing_override", None)
    return resolve_effective_billing_intent(
        transport_request.billing_intent,
        override,
    )


def _destination_label(leg: TransportRequestLeg) -> str:
    establishment = (getattr(leg, "dropoff_establishment", None) or "").strip()
    if establishment:
        return establishment
    doctor = (getattr(leg, "dropoff_doctor", None) or "").strip()
    if doctor:
        return doctor
    location = (leg.dropoff_location or "").strip()
    if getattr(leg, "is_return_stop", False):
        return "Retour institution" if location else "Retour"
    return location or f"Destination {leg.route_sequence_number}"


def build_billing_summary(transport_request: TransportRequest) -> dict[str, Any]:
    """Résumé facturation pour l'UI institution (payeur principal + exceptions)."""
    primary = (transport_request.billing_intent or "patient").lower()
    legs = sorted(
        getattr(transport_request, "legs", None) or [],
        key=lambda item: item.sequence_index,
    )

    effective_intents: list[str] = []
    exceptions: list[dict[str, Any]] = []

    for leg in legs:
        override = getattr(leg, "destination_billing_override", None)
        effective = resolve_effective_billing_intent(primary, override)
        effective_intents.append(effective)

        if override:
            exceptions.append(
                {
                    "destination_label": _destination_label(leg),
                    "dropoff_location": leg.dropoff_location,
                    "destination_billing_override": override,
                    "override_label": billing_intent_label(override),
                    "effective_billing_intent": effective,
                    "effective_label": billing_intent_label(effective),
                    "is_return_stop": bool(getattr(leg, "is_return_stop", False)),
                }
            )

    unique_payers = set(effective_intents)
    payer_count = len(unique_payers)

    return {
        "primary_intent": primary,
        "primary_label": billing_intent_label(primary),
        "multi_payer": payer_count > 1,
        "payer_count": payer_count,
        "has_exceptions": bool(exceptions),
        "exceptions": exceptions,
    }


def billed_to_type_from_intent(billing_intent: str) -> str:
    """Mappe billing_intent vers billed_to_type legacy Booking."""
    intent = (billing_intent or "patient").lower()
    if intent == "institution":
        return "clinic"
    if intent in ("curator", "spc", "insurance", "other"):
        return intent
    return "patient"
