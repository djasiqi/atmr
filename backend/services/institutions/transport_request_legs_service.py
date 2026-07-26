"""Gestion des legs multi-étapes pour TransportRequest (PR5 V1)."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from ext import db
from models.transport_request_leg import TransportRequestLeg
from shared.time_utils import normalize_mission_wall_clock

logger = logging.getLogger(__name__)


def is_multi_stop_enabled() -> bool:
    return os.getenv("INSTITUTION_MULTI_STOP_ENABLED", "false").lower() in (
        "1",
        "true",
        "yes",
    )


@dataclass(frozen=True, slots=True)
class LegStop:
    """Étape intermédiaire (destination, hors retour institution)."""

    dropoff_location: str
    dropoff_lat: float | None = None
    dropoff_lng: float | None = None
    scheduled_time: Any = None
    time_confirmed: bool = False
    dropoff_establishment: str | None = None
    dropoff_service: str | None = None
    dropoff_doctor: str | None = None
    destination_billing_override: str | None = None
    is_return_stop: bool = False


def new_route_group_id() -> str:
    return str(uuid.uuid4())


def build_legs_chain(
    *,
    origin_location: str,
    origin_lat: float | None,
    origin_lng: float | None,
    stops: list[LegStop],
    return_to_institution: bool,
    institution_return_location: str | None = None,
    institution_return_lat: float | None = None,
    institution_return_lng: float | None = None,
    return_scheduled_time: Any = None,
    return_time_confirmed: bool = False,
    return_stop: LegStop | None = None,
) -> list[dict[str, Any]]:
    """Construit une chaîne continue de legs (pickup[i+1] = dropoff[i])."""
    if not stops and not return_to_institution:
        return []

    legs: list[dict[str, Any]] = []
    cur_pick = origin_location
    cur_plat, cur_plng = origin_lat, origin_lng

    waypoints = list(stops)
    if return_to_institution:
        ret_loc = institution_return_location or origin_location
        if return_stop is not None:
            waypoints.append(return_stop)
        else:
            waypoints.append(
                LegStop(
                    dropoff_location=ret_loc,
                    dropoff_lat=(
                        institution_return_lat
                        if institution_return_lat is not None
                        else origin_lat
                    ),
                    dropoff_lng=(
                        institution_return_lng
                        if institution_return_lng is not None
                        else origin_lng
                    ),
                    scheduled_time=return_scheduled_time,
                    time_confirmed=return_time_confirmed,
                    is_return_stop=True,
                )
            )

    for idx, stop in enumerate(waypoints):
        is_return = bool(getattr(stop, "is_return_stop", False))
        legs.append(
            {
                "sequence_index": idx,
                "route_sequence_number": idx + 1,
                "pickup_location": cur_pick,
                "pickup_lat": cur_plat,
                "pickup_lng": cur_plng,
                "dropoff_location": stop.dropoff_location,
                "dropoff_lat": stop.dropoff_lat,
                "dropoff_lng": stop.dropoff_lng,
                "dropoff_establishment": stop.dropoff_establishment,
                "dropoff_service": stop.dropoff_service,
                "dropoff_doctor": stop.dropoff_doctor,
                "scheduled_time": parse_leg_scheduled_time(stop.scheduled_time),
                "time_confirmed": bool(stop.time_confirmed),
                "destination_billing_override": stop.destination_billing_override,
                "is_return_stop": is_return,
            }
        )
        cur_pick = stop.dropoff_location
        cur_plat = stop.dropoff_lat
        cur_plng = stop.dropoff_lng
    return legs


def parse_leg_scheduled_time(value: Any) -> Any:
    """Convertit une ISO API en datetime naïf Genève pour la colonne timestamptz."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        return normalize_mission_wall_clock(stripped)
    if isinstance(value, datetime):
        return normalize_mission_wall_clock(value)
    return value


def remove_stop_at_index(stops: list[LegStop], remove_index: int) -> list[LegStop]:
    """Retire une étape intermédiaire (recalcul chaîne côté appelant)."""
    if remove_index < 0 or remove_index >= len(stops):
        raise ValueError("Index d'étape invalide")
    return [s for i, s in enumerate(stops) if i != remove_index]


def persist_legs(
    transport_request_id: int, legs_data: list[dict[str, Any]]
) -> list[TransportRequestLeg]:
    """Remplace les legs d'une demande (DRAFT/SENT uniquement — vérifié par l'appelant)."""
    TransportRequestLeg.query.filter_by(
        transport_request_id=transport_request_id
    ).delete()
    result: list[TransportRequestLeg] = []
    for ld in legs_data:
        leg = TransportRequestLeg()
        leg.transport_request_id = transport_request_id
        leg.sequence_index = int(ld["sequence_index"])
        leg.route_sequence_number = int(ld["route_sequence_number"])
        leg.pickup_location = ld["pickup_location"]
        leg.pickup_lat = (
            Decimal(str(ld["pickup_lat"])) if ld.get("pickup_lat") is not None else None
        )
        leg.pickup_lng = (
            Decimal(str(ld["pickup_lng"])) if ld.get("pickup_lng") is not None else None
        )
        leg.dropoff_location = ld["dropoff_location"]
        leg.dropoff_lat = (
            Decimal(str(ld["dropoff_lat"]))
            if ld.get("dropoff_lat") is not None
            else None
        )
        leg.dropoff_lng = (
            Decimal(str(ld["dropoff_lng"]))
            if ld.get("dropoff_lng") is not None
            else None
        )
        leg.dropoff_establishment = ld.get("dropoff_establishment") or None
        leg.dropoff_service = ld.get("dropoff_service") or None
        leg.dropoff_doctor = ld.get("dropoff_doctor") or None
        leg.scheduled_time = parse_leg_scheduled_time(ld.get("scheduled_time"))
        leg.time_confirmed = bool(ld.get("time_confirmed", False))
        leg.destination_billing_override = (
            ld.get("destination_billing_override") or None
        )
        leg.is_return_stop = bool(ld.get("is_return_stop", False))
        db.session.add(leg)
        result.append(leg)
    db.session.flush()
    return result


def _parse_destination_billing_override(item: dict[str, Any]) -> str | None:
    if not bool(item.get("use_custom_billing", False)):
        return None
    override = (item.get("destination_billing_override") or "").strip()
    return override.lower() if override else None


def return_stop_from_validated(
    validated: dict[str, Any],
    *,
    return_location: str,
    return_lat: float | None,
    return_lng: float | None,
    return_scheduled_time: Any = None,
    return_time_confirmed: bool = False,
) -> LegStop | None:
    """Construit le stop système retour depuis return_stop ou défaut."""
    raw = validated.get("return_stop")
    override = None
    if isinstance(raw, dict):
        override = _parse_destination_billing_override(raw)
    return LegStop(
        dropoff_location=return_location,
        dropoff_lat=return_lat,
        dropoff_lng=return_lng,
        scheduled_time=return_scheduled_time,
        time_confirmed=return_time_confirmed,
        destination_billing_override=override,
        is_return_stop=True,
    )


def stops_from_validated(validated: dict[str, Any]) -> list[LegStop]:
    """Parse intermediate_stops depuis le payload API."""
    raw = validated.get("intermediate_stops") or []
    items: list[dict[str, Any]] = [item for item in raw if isinstance(item, dict)]
    if any(item.get("sequence") is not None for item in items):
        items = sorted(items, key=lambda item: item.get("sequence") or 0)
    stops: list[LegStop] = []
    for item in items:
        loc = (item.get("dropoff_location") or "").strip()
        if not loc:
            continue
        time_confirmed = bool(item.get("time_confirmed", False))
        raw_time = item.get("scheduled_time")
        parsed_time = parse_leg_scheduled_time(raw_time) if raw_time else None
        if time_confirmed and parsed_time is None and raw_time:
            parsed_time = parse_leg_scheduled_time(str(raw_time))
        if time_confirmed and parsed_time is None:
            raise ValueError(
                "time_confirmed=true requiert scheduled_time sur une étape."
            )
        stops.append(
            LegStop(
                dropoff_location=loc,
                dropoff_lat=item.get("dropoff_lat"),
                dropoff_lng=item.get("dropoff_lng"),
                scheduled_time=parsed_time if parsed_time is not None else raw_time,
                time_confirmed=time_confirmed,
                dropoff_establishment=(item.get("dropoff_establishment") or None),
                dropoff_service=(item.get("dropoff_service") or None),
                dropoff_doctor=(item.get("dropoff_doctor") or None),
                destination_billing_override=_parse_destination_billing_override(item),
                is_return_stop=bool(item.get("is_return_stop", False)),
            )
        )
    return stops


def build_simple_trip_leg(
    *,
    pickup_location: str,
    pickup_lat: float | None,
    pickup_lng: float | None,
    dropoff_location: str,
    dropoff_lat: float | None,
    dropoff_lng: float | None,
    appointment_time: Any,
    time_confirmed: bool,
    dropoff_establishment: str | None = None,
    dropoff_service: str | None = None,
    dropoff_doctor: str | None = None,
) -> list[dict[str, Any]]:
    """Leg unique pour trajet simple avec RDV à la destination."""
    if appointment_time is None and not time_confirmed:
        return []
    return build_legs_chain(
        origin_location=pickup_location,
        origin_lat=pickup_lat,
        origin_lng=pickup_lng,
        stops=[
            LegStop(
                dropoff_location=dropoff_location,
                dropoff_lat=dropoff_lat,
                dropoff_lng=dropoff_lng,
                scheduled_time=appointment_time,
                time_confirmed=time_confirmed,
                dropoff_establishment=dropoff_establishment,
                dropoff_service=dropoff_service,
                dropoff_doctor=dropoff_doctor,
            )
        ],
        return_to_institution=False,
    )


def sync_return_fields_from_legs(transport_request: Any) -> None:
    """Synchronise return_date/time/confirmed depuis le dernier leg (retour)."""
    if not getattr(transport_request, "return_to_institution", False):
        return
    legs = sorted(
        getattr(transport_request, "legs", None) or [],
        key=lambda leg: leg.sequence_index,
    )
    if not legs:
        return
    last = legs[-1]
    mission_day = getattr(transport_request, "mission_date", None)
    if mission_day is not None:
        transport_request.return_date = mission_day
    if last.time_confirmed and last.scheduled_time is not None:
        transport_request.return_time = last.scheduled_time
        transport_request.return_time_confirmed = True
    else:
        transport_request.return_time = None
        transport_request.return_time_confirmed = False


def legs_snapshot(legs: list[TransportRequestLeg]) -> list[dict[str, Any]]:
    """Snapshot compact pour comparaison / timeline."""
    return [
        {
            "sequence_index": leg.sequence_index,
            "route_sequence_number": leg.route_sequence_number,
            "pickup_location": leg.pickup_location,
            "dropoff_location": leg.dropoff_location,
            "scheduled_time": leg.scheduled_time.isoformat()
            if leg.scheduled_time
            else None,
            "time_confirmed": bool(leg.time_confirmed),
        }
        for leg in sorted(legs, key=lambda item: item.sequence_index)
    ]


def reorganize_multi_stop_legs(
    transport_request: Any,
    *,
    intermediate_stops: list[LegStop],
    return_to_institution: bool,
    return_scheduled_time: Any = None,
    return_time_confirmed: bool = False,
    return_stop: LegStop | None = None,
    actor_user_id: int | None = None,
) -> list[TransportRequestLeg]:
    """Recalcule la chaîne de legs et historise si changement."""
    existing = list(transport_request.legs or [])
    before = legs_snapshot(existing)

    origin = transport_request.pickup_location
    legs_data = build_legs_chain(
        origin_location=origin,
        origin_lat=float(transport_request.pickup_lat)
        if transport_request.pickup_lat is not None
        else None,
        origin_lng=float(transport_request.pickup_lng)
        if transport_request.pickup_lng is not None
        else None,
        stops=intermediate_stops,
        return_to_institution=return_to_institution,
        institution_return_location=origin,
        institution_return_lat=float(transport_request.pickup_lat)
        if transport_request.pickup_lat is not None
        else None,
        institution_return_lng=float(transport_request.pickup_lng)
        if transport_request.pickup_lng is not None
        else None,
        return_scheduled_time=return_scheduled_time,
        return_time_confirmed=return_time_confirmed,
        return_stop=return_stop,
    )
    new_legs = persist_legs(int(transport_request.id), legs_data)
    after = legs_snapshot(new_legs)

    transport_request.return_to_institution = return_to_institution
    if legs_data:
        first = legs_data[0]
        transport_request.dropoff_location = first["dropoff_location"]
        transport_request.dropoff_lat = first.get("dropoff_lat")
        transport_request.dropoff_lng = first.get("dropoff_lng")

    sync_return_fields_from_legs(transport_request)

    if before != after:
        _record_legs_reorganized_timeline(
            transport_request=transport_request,
            before=before,
            after=after,
            actor_user_id=actor_user_id,
        )

    return new_legs


def _record_legs_reorganized_timeline(
    *,
    transport_request: Any,
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    actor_user_id: int | None,
) -> None:
    try:
        from services.institutions.transport_timeline_service import (
            TimelineActor,
            record_event,
            resolve_actor_name,
        )

        record_event(
            "route_legs_reorganized",
            institution_id=int(transport_request.institution_id),
            transport_request_id=int(transport_request.id),
            actor=TimelineActor(
                actor_type="institution_user" if actor_user_id else "system",
                actor_user_id=actor_user_id,
            ),
            payload={
                "before_legs": before,
                "after_legs": after,
                "actor_name": resolve_actor_name(actor_user_id),
                "route_group_id": getattr(transport_request, "route_group_id", None),
            },
            correlation_id=(
                f"route_legs_reorganized:{transport_request.id}:{len(after)}"
            ),
        )
    except Exception as timeline_err:
        logger.warning(
            "[TransportRequestLegs] timeline route_legs_reorganized failed: %s",
            timeline_err,
        )
