"""Règles métier acceptation offres institution (Valider / Planifier / Départ immédiat).

Institution = exprime le besoin ; transporteur = définit l'heure opérationnelle (pickup).
Planifier ≡ accept(proposed_pickup_time) — pas de validation institution intermédiaire.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from models.enums import ScheduledTimeType
from services.institutions.mission_schedule import is_operational_time
from shared.time_utils import now_local, parse_iso8601

if TYPE_CHECKING:
    from models.transport_request import TransportRequest


def has_confirmed_departure(transport_request: TransportRequest | Any) -> bool:
    """Départ institutionnel confirmé (scheduled_time = départ)."""
    st_type = getattr(transport_request, "scheduled_time_type", None) or "departure"
    return bool(
        getattr(transport_request, "pickup_time_confirmed", False)
        and getattr(transport_request, "scheduled_time", None)
        and st_type != ScheduledTimeType.ARRIVAL.value
    )


def _parse_wall_clock(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    if isinstance(value, str):
        parsed = parse_iso8601(value)
        if parsed is None:
            return None
        return parsed.replace(tzinfo=None)
    return None


def departure_datetime(transport_request: TransportRequest | Any) -> datetime | None:
    if not has_confirmed_departure(transport_request):
        return None
    return _parse_wall_clock(getattr(transport_request, "scheduled_time", None))


def is_departure_stale(
    transport_request: TransportRequest | Any,
    now: datetime | None = None,
) -> bool:
    """True si le départ confirmé est dans le passé."""
    dep = departure_datetime(transport_request)
    if dep is None:
        return False
    ref = now or now_local()
    ref_naive = ref.replace(tzinfo=None) if ref.tzinfo else ref
    return dep < ref_naive


def has_confirmed_rdv_only(transport_request: TransportRequest | Any) -> bool:
    """RDV confirmé sans départ confirmé."""
    if has_confirmed_departure(transport_request):
        return False

    legs = getattr(transport_request, "legs", None) or []
    for leg in legs:
        if is_operational_time(
            scheduled_time=getattr(leg, "scheduled_time", None),
            time_confirmed=bool(getattr(leg, "time_confirmed", False)),
        ):
            return True

    st_type = getattr(transport_request, "scheduled_time_type", None) or "departure"
    appt_confirmed = getattr(transport_request, "appointment_time_confirmed", None)
    if appt_confirmed is None and st_type == ScheduledTimeType.ARRIVAL.value:
        appt_confirmed = True
    return bool(
        st_type == ScheduledTimeType.ARRIVAL.value
        and is_operational_time(
            scheduled_time=getattr(transport_request, "scheduled_time", None),
            time_confirmed=bool(appt_confirmed),
        )
    )


def has_any_operational_schedule(transport_request: TransportRequest | Any) -> bool:
    return has_confirmed_departure(transport_request) or has_confirmed_rdv_only(
        transport_request
    )


def can_validate_without_proposed_pickup(
    transport_request: TransportRequest | Any,
    now: datetime | None = None,
) -> bool:
    """Valider : accept sans proposed_pickup_time."""
    return has_confirmed_departure(transport_request) and not is_departure_stale(
        transport_request, now=now
    )


def validate_accept_pickup_rules(
    transport_request: TransportRequest | Any,
    *,
    proposed_pickup_time: datetime | None,
    now: datetime | None = None,
) -> str | None:
    """Retourne un message d'erreur si accept interdit, sinon None."""
    if proposed_pickup_time is not None:
        return None
    if can_validate_without_proposed_pickup(transport_request, now=now):
        return None
    return (
        "Définissez l'heure de prise en charge (proposed_pickup_time requis). "
        "Utilisez Planifier ou Départ immédiat."
    )
