"""Helpers horaires mission institution — source de vérité opérationnelle.

Invariant :
- ``TransportRequest.scheduled_time`` = heure de départ uniquement (nullable).
- ``TransportRequestLeg.scheduled_time`` = RDV / retour par étape.
- ``get_effective_dispatch_time()`` = calcul opérationnel (heures confirmées, même jour).

Règle d'architecture (écritures mission) :
- Toute écriture d'horaire mission DOIT passer par ``normalize_mission_wall_clock()``.
- ``parse_iso8601()`` est interdit pour les écritures mission (retourne aware).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from shared.time_utils import normalize_mission_wall_clock, parse_iso8601

if TYPE_CHECKING:
    from models.transport_request import TransportRequest


def is_operational_time(*, scheduled_time: Any, time_confirmed: bool) -> bool:
    """True si l'heure peut alimenter SLA, alertes, ETA, pénalités."""
    return time_confirmed is True and scheduled_time is not None


def validate_time_pair(*, scheduled_time: Any, time_confirmed: bool) -> None:
    """time_confirmed=true implique scheduled_time != null."""
    if time_confirmed and scheduled_time is None:
        raise ValueError("time_confirmed=true requiert scheduled_time renseigné.")


def _to_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _same_mission_date(scheduled_time: Any, mission_day: date | None) -> bool:
    if mission_day is None or scheduled_time is None:
        return False
    if isinstance(scheduled_time, datetime):
        return scheduled_time.date() == mission_day
    if isinstance(scheduled_time, str):
        parsed = parse_iso8601(scheduled_time)
        return parsed is not None and parsed.date() == mission_day
    return False


def get_mission_date(transport_request: TransportRequest) -> date | None:
    """Date métier de la mission."""
    raw = getattr(transport_request, "mission_date", None)
    if raw is not None:
        return _to_date(raw)
    st = getattr(transport_request, "scheduled_time", None)
    if st is not None:
        return st.date() if isinstance(st, datetime) else None
    return None


def get_effective_dispatch_time(transport_request: TransportRequest) -> datetime | None:
    """Prochaine heure confirmée chronologique (départ + legs), même ``mission_date``."""
    mission_day = get_mission_date(transport_request)
    candidates: list[datetime] = []

    pickup_confirmed = bool(getattr(transport_request, "pickup_time_confirmed", False))
    dep = getattr(transport_request, "scheduled_time", None)
    if is_operational_time(
        scheduled_time=dep, time_confirmed=pickup_confirmed
    ) and _same_mission_date(dep, mission_day):
        candidates.append(dep)

    legs = sorted(
        getattr(transport_request, "legs", None) or [],
        key=lambda leg: getattr(leg, "sequence_index", 0),
    )
    for leg in legs:
        leg_st = getattr(leg, "scheduled_time", None)
        leg_confirmed = bool(getattr(leg, "time_confirmed", False))
        if is_operational_time(
            scheduled_time=leg_st, time_confirmed=leg_confirmed
        ) and _same_mission_date(leg_st, mission_day):
            candidates.append(leg_st)

    if not candidates:
        return None
    return min(candidates)


def has_at_least_one_confirmed_time(transport_request: TransportRequest) -> bool:
    """Au moins une heure confirmée (départ, leg ou retour)."""
    return get_effective_dispatch_time(transport_request) is not None


def parse_mission_date(validated: dict[str, Any]) -> date:
    """Extrait mission_date du payload (ou depuis scheduled_time legacy)."""
    raw = validated.get("mission_date")
    if raw is not None:
        if isinstance(raw, date) and not isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            return date.fromisoformat(raw[:10])
    st_raw = validated.get("scheduled_time")
    if st_raw:
        parsed = parse_iso8601(str(st_raw))
        if parsed:
            return parsed.date()
    raise ValueError("mission_date est obligatoire.")


def apply_departure_schedule(
    transport_request: TransportRequest, validated: dict[str, Any]
) -> None:
    """Applique mission_date, départ (nullable) et pickup_time_confirmed.

    Ordre d'assignation important : ``scheduled_time`` doit être posé sur le
    modèle AVANT ``pickup_time_confirmed`` car le validator SQLAlchemy
    ``@validates("pickup_time_confirmed")`` exige ``scheduled_time != None``.
    """
    transport_request.mission_date = parse_mission_date(validated)

    pickup_confirmed = validated.get("pickup_time_confirmed")
    if pickup_confirmed is None:
        st_type = validated.get("scheduled_time_type", "departure")
        st_raw = validated.get("scheduled_time")
        pickup_confirmed = bool(st_raw and st_type == "departure")
    pickup_confirmed = bool(pickup_confirmed)

    scheduled_time_type = validated.get("scheduled_time_type", "departure")

    departure_dt: datetime | None = None
    st_raw = validated.get("scheduled_time")
    if st_raw and scheduled_time_type == "departure":
        departure_dt = normalize_mission_wall_clock(st_raw)
        if pickup_confirmed:
            validate_time_pair(scheduled_time=departure_dt, time_confirmed=True)

    # scheduled_time AVANT pickup_time_confirmed (invariant validator modèle).
    transport_request.scheduled_time = departure_dt
    transport_request.scheduled_time_type = scheduled_time_type
    transport_request.pickup_time_confirmed = pickup_confirmed


def legacy_arrival_datetime(validated: dict[str, Any]) -> datetime | None:
    """RDV unique legacy (type=arrival) sans multi-stop."""
    dt, _confirmed = legacy_arrival_schedule(validated)
    return dt


def legacy_arrival_schedule(
    validated: dict[str, Any],
) -> tuple[datetime | None, bool]:
    """RDV legacy : (datetime, time_confirmed)."""
    if validated.get("multi_stop"):
        return None, False
    if validated.get("pickup_time_confirmed"):
        return None, False
    st_type = validated.get("scheduled_time_type", "departure")
    if st_type != "arrival":
        return None, False
    raw = validated.get("scheduled_time")
    if not raw:
        return None, False
    dt = normalize_mission_wall_clock(raw)
    confirmed = validated.get("appointment_time_confirmed")
    if confirmed is None:
        confirmed = True
    if confirmed and dt is None:
        raise ValueError("appointment_time_confirmed=true requiert scheduled_time.")
    return dt, bool(confirmed)


def sync_transport_request_departure_from_booking(
    transport_request: TransportRequest | None,
    booking: Any,
) -> bool:
    """Aligne le départ request sur le booking principal (source opérationnelle)."""
    if transport_request is None or booking is None:
        return False
    if getattr(transport_request, "booking_id", None) != getattr(booking, "id", None):
        return False
    scheduled = getattr(booking, "scheduled_time", None)
    if scheduled is None:
        return False

    from models.enums import ScheduledTimeType

    transport_request.scheduled_time = scheduled
    transport_request.scheduled_time_type = ScheduledTimeType.DEPARTURE.value
    transport_request.pickup_time_confirmed = True
    return True


def sync_request_departure_for_booking(booking: Any) -> bool:
    """Aligne transport_request.scheduled_time depuis le booking principal lié."""
    if booking is None or getattr(booking, "id", None) is None:
        return False
    from models.transport_request import TransportRequest

    transport_request = TransportRequest.query.filter_by(booking_id=booking.id).first()
    return sync_transport_request_departure_from_booking(transport_request, booking)
