"""Invariant temporel A/R : un retour confirmé ne peut pas précéder l'aller.

Règles :
- ``return_pickup > outbound_pickup`` (minimum absolu)
- si un RDV / contrainte destination existe : ``return_pickup >= appointment``
- un changement d'horaire amont n'annule pas la mission
- un retour encore chronologiquement valide reste confirmé
- un retour devenu impossible perd uniquement ``time_confirmed``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

from shared.time_utils import normalize_mission_wall_clock, parse_local_naive

logger = logging.getLogger(__name__)

TEMPORAL_CONFLICT_CODE = "TEMPORAL_CONFLICT"
RETURN_BEFORE_OUTBOUND_MESSAGE = (
    "L'heure de retour ne peut pas être antérieure à l'aller."
)
RETURN_BEFORE_APPOINTMENT_MESSAGE = (
    "L'heure de retour ne peut pas précéder l'heure de rendez-vous."
)


@dataclass(frozen=True, slots=True)
class TemporalConflict:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_error_dict(self) -> dict[str, str]:
        return {
            "error": self.code,
            "message": self.message,
        }


def as_naive_datetime(value: Any) -> datetime | None:
    """Datetime naïf minute près, ou None."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value.replace(tzinfo=None) if value.tzinfo else value
    else:
        parsed = normalize_mission_wall_clock(value)
        if parsed is None:
            try:
                parsed = parse_local_naive(value)
            except Exception:
                return None
    if parsed is None:
        return None
    return parsed.replace(second=0, microsecond=0)


def _hhmm(value: Any) -> str | None:
    dt = as_naive_datetime(value)
    if dt is None:
        return None
    return f"{dt.hour:02d}:{dt.minute:02d}"


def booking_pickup_datetime(booking: Any) -> datetime | None:
    """Heure de prise en charge stockée (indépendamment de la confirmation)."""
    return as_naive_datetime(getattr(booking, "scheduled_time", None))


def confirmed_pickup_datetime(booking: Any) -> datetime | None:
    if not bool(getattr(booking, "time_confirmed", False)):
        return None
    return booking_pickup_datetime(booking)


def is_return_leg(booking: Any) -> bool:
    if bool(getattr(booking, "is_return", False)):
        return True
    if bool(getattr(booking, "_is_return_leg_from_topology", False)):
        return True
    seq = getattr(booking, "route_sequence_number", None)
    try:
        return seq is not None and int(seq) > 1
    except (TypeError, ValueError):
        return False


def resolve_destination_constraint(
    outbound: Any,
    *,
    appointment_dt: Any = None,
) -> datetime | None:
    """Borne minimale renforcée : RDV / destination de l'aller."""
    explicit = as_naive_datetime(appointment_dt)
    if explicit is not None:
        return explicit

    brief_fn = getattr(outbound, "_get_institution_leg_clinical_brief", None)
    if callable(brief_fn):
        try:
            brief = brief_fn()
        except Exception:
            brief = None
        if isinstance(brief, dict):
            from_brief = as_naive_datetime(brief.get("appointment_time"))
            if from_brief is not None:
                return from_brief

    req = _resolve_source_request(outbound)
    if req is None:
        return None
    legs = list(getattr(req, "legs", None) or [])
    legs.sort(key=lambda item: getattr(item, "sequence_index", 0) or 0)
    dest_legs = legs
    if bool(getattr(req, "return_to_institution", False)) and len(legs) > 1:
        dest_legs = legs[:-1]
    latest: datetime | None = None
    for leg in dest_legs:
        candidate = as_naive_datetime(getattr(leg, "scheduled_time", None))
        if candidate is None:
            continue
        if latest is None or candidate > latest:
            latest = candidate
    return latest


def evaluate_return_chronology(
    *,
    outbound_pickup: Any,
    return_pickup: Any,
    appointment_dt: Any = None,
) -> TemporalConflict | None:
    """Vérifie qu'un horaire retour est possible par rapport à l'aller."""
    ret_dt = as_naive_datetime(return_pickup)
    if ret_dt is None:
        return None

    out_dt = as_naive_datetime(outbound_pickup)
    dest_dt = as_naive_datetime(appointment_dt)

    if out_dt is not None and ret_dt <= out_dt:
        return TemporalConflict(
            code=TEMPORAL_CONFLICT_CODE,
            message=RETURN_BEFORE_OUTBOUND_MESSAGE,
            details={
                "outbound_pickup": _hhmm(out_dt),
                "return_pickup": _hhmm(ret_dt),
            },
        )
    if dest_dt is not None and ret_dt < dest_dt:
        return TemporalConflict(
            code=TEMPORAL_CONFLICT_CODE,
            message=RETURN_BEFORE_APPOINTMENT_MESSAGE,
            details={
                "appointment": _hhmm(dest_dt),
                "return_pickup": _hhmm(ret_dt),
            },
        )
    return None


def validate_confirmed_return_pickup(
    return_booking: Any,
    *,
    intended_pickup: Any = None,
    outbound: Any = None,
    appointment_dt: Any = None,
) -> TemporalConflict | None:
    """Rejette une confirmation de retour chronologiquement impossible."""
    pickup = (
        intended_pickup
        if intended_pickup is not None
        else getattr(return_booking, "scheduled_time", None)
    )
    parent = outbound if outbound is not None else load_predecessor(return_booking)
    if parent is None:
        return None
    dest = resolve_destination_constraint(parent, appointment_dt=appointment_dt)
    return evaluate_return_chronology(
        outbound_pickup=booking_pickup_datetime(parent),
        return_pickup=pickup,
        appointment_dt=dest,
    )


def load_predecessor(booking: Any) -> Any | None:
    """Aller / leg précédent dont dépend temporellement ``booking``."""
    parent = load_outbound(booking)
    if parent is not None:
        return parent
    return load_previous_route_leg(booking)


def load_previous_route_leg(booking: Any) -> Any | None:
    cached = getattr(booking, "_previous_leg", None)
    if cached is not None:
        return cached
    group_id = getattr(booking, "route_group_id", None)
    seq = getattr(booking, "route_sequence_number", None)
    if not group_id or seq is None:
        return None
    try:
        seq_n = int(seq)
    except (TypeError, ValueError):
        return None
    if seq_n <= 1:
        return None
    try:
        from models.booking import Booking

        return (
            Booking.query.filter(
                Booking.route_group_id == group_id,
                Booking.route_sequence_number < seq_n,
                Booking.id != getattr(booking, "id", None),
            )
            .order_by(Booking.route_sequence_number.desc(), Booking.id.desc())
            .first()
        )
    except Exception:
        return None


def load_outbound(booking: Any) -> Any | None:
    """Aller lié à un retour (parent_booking_id)."""
    if booking is None:
        return None
    parent_id = getattr(booking, "parent_booking_id", None)
    if not parent_id and not is_return_leg(booking):
        return None

    related = getattr(booking, "return_trip", None)
    if related is not None and getattr(related, "id", None) == parent_id:
        return related
    original = getattr(booking, "original_booking", None)
    if (
        original is not None
        and parent_id
        and getattr(original, "id", None) == parent_id
    ):
        return original

    if not parent_id:
        return None
    try:
        from models.booking import Booking

        return Booking.query.get(parent_id)
    except Exception:
        return None


def load_downstream_returns(booking: Any) -> list[Any]:
    """Retours (et legs aval) dépendant temporellement de ``booking``."""
    if booking is None:
        return []

    cached: list[Any] = []
    original = getattr(booking, "original_booking", None)
    if original is not None and getattr(original, "parent_booking_id", None) == getattr(
        booking, "id", None
    ):
        cached.append(original)
    explicit = getattr(booking, "_downstream_returns", None)
    if explicit:
        cached.extend(list(explicit))
    if cached:
        return _unique_bookings(cached)

    results: list[Any] = []
    booking_id = getattr(booking, "id", None)
    try:
        from models.booking import Booking

        if booking_id is not None:
            results.extend(
                Booking.query.filter_by(parent_booking_id=booking_id)
                .order_by(Booking.id.asc())
                .all()
            )
        group_id = getattr(booking, "route_group_id", None)
        seq = getattr(booking, "route_sequence_number", None)
        if group_id and seq is not None and not is_return_leg(booking):
            later = (
                Booking.query.filter(
                    Booking.route_group_id == group_id,
                    Booking.route_sequence_number > seq,
                    Booking.id != booking_id,
                )
                .order_by(Booking.route_sequence_number.asc(), Booking.id.asc())
                .all()
            )
            for later_leg in later:
                results.append(later_leg)
                results.extend(
                    Booking.query.filter_by(parent_booking_id=later_leg.id)
                    .order_by(Booking.id.asc())
                    .all()
                )
    except Exception:
        logger.debug(
            "[RoundTripTemporal] load_downstream_returns fallback booking=%s",
            booking_id,
            exc_info=True,
        )
    return _unique_bookings(results)


def invalidate_impossible_downstream_confirmations(
    booking: Any,
    *,
    appointment_dt: Any = None,
    downstream: Iterable[Any] | None = None,
) -> list[str]:
    """Retire la confirmation des legs aval devenus impossibles.

    Ne change pas ``booking.status``. Ne fabrique pas de nouvel horaire.
    Conserve ``scheduled_time`` pour l'audit / l'historique.
    """
    updated: list[str] = []
    siblings = (
        list(downstream) if downstream is not None else load_downstream_returns(booking)
    )
    if not siblings:
        return updated

    outbound_pickup = booking_pickup_datetime(booking)
    dest = resolve_destination_constraint(booking, appointment_dt=appointment_dt)
    if outbound_pickup is None and dest is None:
        return updated

    for sibling in siblings:
        if not bool(getattr(sibling, "time_confirmed", False)):
            continue
        sibling_pickup = booking_pickup_datetime(sibling)
        if sibling_pickup is None:
            continue
        conflict = evaluate_return_chronology(
            outbound_pickup=outbound_pickup,
            return_pickup=sibling_pickup,
            appointment_dt=dest,
        )
        if conflict is None:
            continue
        previous = getattr(sibling, "scheduled_time", None)
        sibling.time_confirmed = False
        _sync_request_return_unconfirmed(sibling)
        _record_return_invalidation(
            sibling,
            previous_time=previous,
            reason=conflict.message,
            upstream_id=getattr(booking, "id", None),
        )
        sibling_id = getattr(sibling, "id", None)
        updated.append(f"return[{sibling_id}].time_confirmed")
        logger.info(
            "[RoundTripTemporal] invalidation retour #%s (amont=#%s) : %s",
            sibling_id,
            getattr(booking, "id", None),
            conflict.message,
        )
    return updated


def apply_round_trip_schedule_rules(
    booking: Any,
    *,
    intended_pickup: Any = None,
    intended_confirmed: bool | None = None,
    appointment_dt: Any = None,
) -> TemporalConflict | None:
    """Valide une confirmation de retour avant écriture."""
    confirmed = (
        bool(intended_confirmed)
        if intended_confirmed is not None
        else bool(getattr(booking, "time_confirmed", False))
    )
    if not confirmed:
        return None
    pickup = (
        intended_pickup
        if intended_pickup is not None
        else getattr(booking, "scheduled_time", None)
    )
    return validate_confirmed_return_pickup(
        booking,
        intended_pickup=pickup,
        appointment_dt=appointment_dt,
    )


def audit_impossible_round_trips(
    *,
    repair: bool = False,
) -> dict[str, Any]:
    """Compte (et optionnellement répare) les A/R à chronologie impossible."""
    from models.booking import Booking
    from models.enums import BookingStatus

    excluded = {
        BookingStatus.CANCELED.value,
        "CANCELLED",
        "REJECTED",
    }
    returns = Booking.query.filter(
        Booking.is_return.is_(True),
        Booking.parent_booking_id.isnot(None),
        Booking.time_confirmed.is_(True),
        Booking.scheduled_time.isnot(None),
    ).all()
    grouped_legs = Booking.query.filter(
        Booking.route_group_id.isnot(None),
        Booking.time_confirmed.is_(True),
        Booking.scheduled_time.isnot(None),
    ).all()

    candidates: list[tuple[Any, Any]] = []
    seen: set[int] = set()
    for ret in returns:
        outbound = load_outbound(ret)
        if outbound is None:
            continue
        candidates.append((outbound, ret))
        if ret.id is not None:
            seen.add(int(ret.id))

    by_group: dict[Any, list[Any]] = {}
    for leg in grouped_legs:
        by_group.setdefault(leg.route_group_id, []).append(leg)
    for legs in by_group.values():
        legs.sort(
            key=lambda item: (
                getattr(item, "route_sequence_number", 0) or 0,
                getattr(item, "id", 0) or 0,
            )
        )
        for idx in range(1, len(legs)):
            nxt = legs[idx]
            if nxt.id is not None and int(nxt.id) in seen:
                continue
            prev = legs[idx - 1]
            candidates.append((prev, nxt))
            if nxt.id is not None:
                seen.add(int(nxt.id))

    pairs: list[dict[str, Any]] = []
    repaired_ids: list[int] = []
    for outbound, ret in candidates:
        status = getattr(
            getattr(ret, "status", None), "value", getattr(ret, "status", None)
        )
        if str(status or "").upper() in {str(item).upper() for item in excluded}:
            continue
        dest = resolve_destination_constraint(outbound)
        conflict = evaluate_return_chronology(
            outbound_pickup=booking_pickup_datetime(outbound),
            return_pickup=booking_pickup_datetime(ret),
            appointment_dt=dest,
        )
        if conflict is None:
            continue
        row = {
            "return_id": ret.id,
            "outbound_id": getattr(outbound, "id", None),
            "customer_name": getattr(ret, "customer_name", None),
            "outbound_pickup": _hhmm(outbound.scheduled_time),
            "return_pickup": _hhmm(ret.scheduled_time),
            "appointment": _hhmm(dest),
            "reason": conflict.message,
        }
        pairs.append(row)
        if repair:
            ret.time_confirmed = False
            _sync_request_return_unconfirmed(ret)
            _record_return_invalidation(
                ret,
                previous_time=getattr(ret, "scheduled_time", None),
                reason=conflict.message,
                upstream_id=getattr(outbound, "id", None),
            )
            repaired_ids.append(int(ret.id))

    return {
        "count": len(pairs),
        "ids": [row["return_id"] for row in pairs],
        "pairs": pairs,
        "repaired_ids": repaired_ids,
    }


def _unique_bookings(bookings: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    unique: list[Any] = []
    for item in bookings:
        key = getattr(item, "id", id(item))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _resolve_source_request(booking: Any) -> Any | None:
    resolver = getattr(booking, "_resolve_source_transport_request", None)
    if callable(resolver):
        try:
            return resolver()
        except Exception:
            return None
    reqs = getattr(booking, "source_request", None)
    if not reqs:
        return None
    return reqs[0] if isinstance(reqs, list) else reqs


def _sync_request_return_unconfirmed(return_booking: Any) -> None:
    outbound = load_outbound(return_booking)
    req = _resolve_source_request(outbound or return_booking)
    if req is None:
        return
    if getattr(req, "return_time_confirmed", False):
        req.return_time_confirmed = False


def _record_return_invalidation(
    return_booking: Any,
    *,
    previous_time: Any,
    reason: str,
    upstream_id: Any,
) -> None:
    """Conserve l'ancien horaire dans l'audit institution si possible."""
    booking_id = getattr(return_booking, "id", None)
    if booking_id is None or not hasattr(return_booking, "__tablename__"):
        return
    try:
        from models.booking_change_event import BookingChangeEvent

        event = BookingChangeEvent(
            booking_id=int(booking_id),
            booking_version=int(getattr(return_booking, "edit_version", None) or 1),
            actor_type="system",
            actor_display_name="Invariant temporel A/R",
            action_type="return_time_invalidated",
            change_class="minor",
            severity="INFO",
            before_snapshot={
                "scheduled_time": str(previous_time) if previous_time else None,
                "time_confirmed": True,
            },
            after_snapshot={
                "scheduled_time": str(previous_time) if previous_time else None,
                "time_confirmed": False,
            },
            changed_fields={
                "time_confirmed": {"from": True, "to": False},
            },
            reason=reason,
            change_scope="schedule",
            source="system",
            operational_impact={
                "upstream_booking_id": upstream_id,
                "status_preserved": True,
            },
        )
        from ext import db

        db.session.add(event)
    except Exception:
        logger.debug(
            "[RoundTripTemporal] audit event skip booking=%s",
            booking_id,
            exc_info=True,
        )
