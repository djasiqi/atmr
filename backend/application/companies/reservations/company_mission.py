"""Contrat canonique de mission entreprise : étapes, ancre, tarif, idempotence."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from sqlalchemy.exc import IntegrityError

from ext import db
from models import Booking
from models.company_booking_mission import (
    CompanyBookingRouteStep,
    CompanyManualBookingRequest,
    CompanyManualBookingRequestOccurrence,
)
from models.enums import BillingSource, BookingCreatedVia, BookingStatus
from shared.time_utils import mission_scheduled_to_api_iso, parse_local_naive

logger = logging.getLogger(__name__)

_CONTRACT_PUT_FIELDS = frozenset(
    {
        "pickup_location",
        "dropoff_location",
        "scheduled_time",
        "amount",
        "billed_to_type",
        "billed_to_company_id",
        "billed_to_contact",
        "mission_type",
        "delivery_description",
        "is_urgent",
        "wheelchair_client_has",
        "wheelchair_need",
        "needs_assistance",
        "passenger_name",
        "requester_name",
        "requester_phone",
        "requester_service",
        "pricing_mode",
        "preferential_amount",
        "is_round_trip",
        "return_time",
        "return_date",
        "pickup_lat",
        "pickup_lon",
        "dropoff_lat",
        "dropoff_lon",
    }
)

_LEGACY_STRUCTURAL = frozenset(
    {
        "pickup_location",
        "dropoff_location",
        "scheduled_time",
        "is_round_trip",
        "return_date",
        "return_time",
    }
)


class CompanyMissionError(Exception):
    def __init__(self, message: str, status_code: int = 400, *, error_code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    """SHA-256 du DTO après normalisation métier, clé d'idempotence exclue."""
    body = {k: v for k, v in payload.items() if k != "idempotency_key"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _iso_dt(value: datetime | None) -> str | None:
    return mission_scheduled_to_api_iso(value)


def _coord(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def _stable_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "position": int(step["position"]),
        "kind": step["kind"],
        "location": step["location"],
        "latitude": _coord(step.get("latitude")),
        "longitude": _coord(step.get("longitude")),
        "arrival_at": _iso_dt(step.get("arrival_at")),
        "departure_at": _iso_dt(step.get("departure_at")),
        "destination_kind": step.get("destination_kind"),
        "establishment": step.get("establishment"),
        "service": step.get("service"),
        "doctor": step.get("doctor"),
        "access_notes": step.get("access_notes"),
    }


def _stable_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _iso_dt(value)
    if isinstance(value, Decimal):
        return f"{_money(value):.2f}"
    if isinstance(value, list):
        return [_stable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _stable_value(item) for key, item in value.items()}
    return value


def hash_canonical_request(validated_data: dict[str, Any]) -> str:
    """Hash stable : retour dérivé, décimaux et horaires normalisés."""
    mission_type = (validated_data.get("mission_type") or "patient_transport").strip().lower()
    steps = normalize_route_steps(
        list(validated_data.get("route_steps") or []),
        mission_type=mission_type,
    )
    body = {
        key: _stable_value(value)
        for key, value in validated_data.items()
        if key not in ("idempotency_key", "route_steps")
    }
    body["route_steps"] = [_stable_step(step) for step in steps]
    amounts = validated_data.get("segment_amounts")
    if amounts:
        body["segment_amounts"] = [
            {
                "from_position": int(row["from_position"]),
                "to_position": int(row["to_position"]),
                "amount": f"{_money(row['amount']):.2f}",
            }
            for row in amounts
        ]
    preferential = validated_data.get("preferential_amount")
    if preferential not in (None, ""):
        body["preferential_amount"] = f"{_money(preferential):.2f}"
    return canonical_payload_hash(body)


def _parse_dt(value: Any, field: str) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    try:
        return parse_local_naive(str(value))
    except Exception as exc:
        raise CompanyMissionError(f"{field} invalide.") from exc


def _money(value: Any) -> Decimal:
    try:
        return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except Exception as exc:
        raise CompanyMissionError("Montant invalide.") from exc


def normalize_route_steps(
    raw_steps: list[dict[str, Any]],
    *,
    mission_type: str,
) -> list[dict[str, Any]]:
    """Valide la forme du parcours et dérive le retour depuis le pickup."""
    if not isinstance(raw_steps, list) or len(raw_steps) < 2:
        raise CompanyMissionError("Un parcours contient au moins deux étapes.")

    steps: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise CompanyMissionError("Étape de parcours invalide.")
        position = raw.get("position", index)
        if int(position) != index:
            raise CompanyMissionError("Les positions doivent être continues à partir de 0.")
        kind = str(raw.get("kind") or "").strip()
        location = str(raw.get("location") or "").strip()
        if not location:
            raise CompanyMissionError("Chaque étape a une adresse.")
        steps.append(
            {
                "position": index,
                "kind": kind,
                "location": location,
                "latitude": raw.get("latitude", raw.get("lat")),
                "longitude": raw.get("longitude", raw.get("lon")),
                "arrival_at": _parse_dt(raw.get("arrival_at"), "arrival_at"),
                "departure_at": _parse_dt(raw.get("departure_at"), "departure_at"),
                "destination_kind": raw.get("destination_kind") or None,
                "establishment": (raw.get("establishment") or None),
                "service": raw.get("service") or None,
                "doctor": raw.get("doctor") or None,
                "access_notes": raw.get("access_notes") or None,
            }
        )

    kinds = [step["kind"] for step in steps]
    if kinds[0] != "pickup" or kinds.count("pickup") != 1:
        raise CompanyMissionError("Le départ est unique et en première position.")
    if kinds.count("destination") < 1:
        raise CompanyMissionError("Le parcours contient au moins une destination.")
    if kinds.count("return") > 1 or ("return" in kinds and kinds[-1] != "return"):
        raise CompanyMissionError("Le retour est unique et en dernière position.")
    middle = kinds[1:-1] if kinds[-1] == "return" else kinds[1:]
    if any(kind != "destination" for kind in middle):
        raise CompanyMissionError("Les étapes intermédiaires sont des destinations.")

    pickup = steps[0]
    if pickup["arrival_at"] is not None or pickup["departure_at"] is None:
        raise CompanyMissionError("Le départ a une heure de départ, sans heure d'arrivée.")

    has_return = kinds[-1] == "return"
    for index, step in enumerate(steps):
        is_last = index == len(steps) - 1
        if step["kind"] == "destination":
            if step["arrival_at"] is None:
                raise CompanyMissionError("Une destination a une heure d'arrivée.")
            followed_by_return = has_return and index == len(steps) - 2
            if (
                not is_last
                and step["departure_at"] is None
                and not followed_by_return
            ):
                raise CompanyMissionError(
                    "Une destination suivie d'une étape a une heure de départ."
                )
            if is_last and step["departure_at"] is not None:
                raise CompanyMissionError("La destination finale n'a pas d'heure de départ.")
            if mission_type == "patient_transport" and step["destination_kind"] not in (
                "medical",
                "other",
            ):
                raise CompanyMissionError(
                    "Le type de lieu est obligatoire pour un transport de personne."
                )
            if mission_type == "material_delivery":
                step["destination_kind"] = None
        if step["kind"] in ("pickup", "return"):
            step["destination_kind"] = None
            step["establishment"] = None
            step["service"] = None
            step["doctor"] = None
        if step["kind"] == "return":
            if step["departure_at"] is not None:
                raise CompanyMissionError("Le retour n'a pas d'heure de départ.")
            previous = steps[index - 1]
            if previous.get("departure_at") is None:
                if step["arrival_at"] is not None:
                    raise CompanyMissionError(
                        "L'heure de retour est à définir : pas d'arrivée tant que le départ n'est pas fixé."
                    )
            elif step["arrival_at"] is None:
                raise CompanyMissionError("Le retour a une heure d'arrivée.")
            step["location"] = pickup["location"]
            step["latitude"] = pickup["latitude"]
            step["longitude"] = pickup["longitude"]

    previous_depart: datetime | None = None
    for step in steps:
        arrival = step["arrival_at"]
        departure = step["departure_at"]
        if arrival is not None and departure is not None and departure < arrival:
            raise CompanyMissionError("L'heure de départ d'une étape précède son arrivée.")
        if previous_depart is not None and arrival is not None and arrival < previous_depart:
            raise CompanyMissionError("Les étapes ne sont pas dans l'ordre chronologique.")
        if departure is not None:
            previous_depart = departure
        elif arrival is not None:
            previous_depart = arrival

    if has_return and steps[-1]["location"] != pickup["location"]:
        raise CompanyMissionError("Le retour doit reprendre le départ.")
    return steps


def is_simple_round_trip(steps: list[dict[str, Any]]) -> bool:
    kinds = [step["kind"] for step in steps]
    return kinds == ["pickup", "destination", "return"]


def resolve_company_mission_anchor(booking: Any) -> Any | None:
    """Ancre canonique depuis n'importe quel segment, ou None si legacy."""
    if booking is None or getattr(booking, "id", None) is None:
        return None
    booking_id = int(booking.id)
    owned = (
        db.session.query(CompanyBookingRouteStep.id)
        .filter(CompanyBookingRouteStep.anchor_booking_id == booking_id)
        .limit(1)
        .first()
    )
    if owned is not None:
        return booking

    parent_id = getattr(booking, "parent_booking_id", None)
    if parent_id:
        parent_owned = (
            db.session.query(CompanyBookingRouteStep.id)
            .filter(CompanyBookingRouteStep.anchor_booking_id == int(parent_id))
            .limit(1)
            .first()
        )
        if parent_owned is not None:
            return db.session.get(Booking, int(parent_id))

    group_id = getattr(booking, "route_group_id", None)
    if group_id:
        anchor_id = (
            db.session.query(CompanyBookingRouteStep.anchor_booking_id)
            .join(Booking, Booking.id == CompanyBookingRouteStep.anchor_booking_id)
            .filter(Booking.route_group_id == group_id)
            .limit(1)
            .scalar()
        )
        if anchor_id is not None:
            return db.session.get(Booking, int(anchor_id))
    return None


def reject_canonical_contract_update(booking: Any, validated_data: dict[str, Any]) -> str | None:
    """Message d'erreur si un PUT toucherait le contrat d'une mission canonique."""
    touched = _CONTRACT_PUT_FIELDS.intersection(validated_data)
    if not touched:
        return None
    if resolve_company_mission_anchor(booking) is None:
        return None
    return (
        "Cette course appartient à une mission canonique. "
        "L'adresse, l'horaire, le tarif et les champs de mission "
        "se modifient avec le parcours, pas sur un segment isolé."
    )


def _steps_for_anchor(anchor_id: int) -> list[CompanyBookingRouteStep]:
    return (
        CompanyBookingRouteStep.query.filter_by(anchor_booking_id=anchor_id)
        .order_by(CompanyBookingRouteStep.position.asc())
        .all()
    )


def _step_payload(step: CompanyBookingRouteStep) -> dict[str, Any]:
    return {
        "position": step.position,
        "kind": step.kind,
        "location": step.location,
        "latitude": float(step.latitude) if step.latitude is not None else None,
        "longitude": float(step.longitude) if step.longitude is not None else None,
        "arrival_at": mission_scheduled_to_api_iso(step.arrival_at),
        "departure_at": mission_scheduled_to_api_iso(step.departure_at),
        "destination_kind": step.destination_kind,
        "establishment": step.establishment,
        "service": step.service,
        "doctor": step.doctor,
        "access_notes": step.access_notes,
    }


def _synthesize_legacy_steps(
    booking: Any, *, company_id: int | None = None
) -> list[dict[str, Any]]:
    base = booking
    parent_id = getattr(booking, "parent_booking_id", None)
    if getattr(booking, "is_return", False) and parent_id:
        if company_id is None:
            parent = db.session.get(Booking, int(parent_id))
        else:
            parent = Booking.query.filter(
                Booking.id == int(parent_id),
                Booking.company_id == company_id,
            ).first()
        if parent is not None:
            base = parent
    steps = [
        {
            "position": 0,
            "kind": "pickup",
            "location": base.pickup_location,
            "latitude": base.pickup_lat,
            "longitude": base.pickup_lon,
            "arrival_at": None,
            "departure_at": mission_scheduled_to_api_iso(base.scheduled_time),
            "destination_kind": None,
            "establishment": None,
            "service": None,
            "doctor": None,
            "access_notes": base.pickup_access_notes,
        },
        {
            "position": 1,
            "kind": "destination",
            "location": base.dropoff_location,
            "latitude": base.dropoff_lat,
            "longitude": base.dropoff_lon,
            "arrival_at": None,
            "departure_at": None,
            "destination_kind": None,
            "establishment": base.medical_facility,
            "service": base.hospital_service,
            "doctor": base.doctor_name,
            "access_notes": base.dropoff_access_notes,
        },
    ]
    child_query = Booking.query.filter_by(parent_booking_id=base.id, is_return=True)
    if company_id is not None:
        child_query = child_query.filter(Booking.company_id == company_id)
    child = child_query.order_by(Booking.id.asc()).first()
    if child is not None:
        steps.append(
            {
                "position": 2,
                "kind": "return",
                "location": child.dropoff_location,
                "latitude": child.dropoff_lat,
                "longitude": child.dropoff_lon,
                "arrival_at": None,
                "departure_at": mission_scheduled_to_api_iso(child.scheduled_time),
                "destination_kind": None,
                "establishment": None,
                "service": None,
                "doctor": None,
                "access_notes": None,
            }
        )
    return steps


def detach_company_mission_anchor(booking_id: int) -> None:
    """Retire l'ancre des occurrences d'idempotence avant suppression du booking.

    La clé étrangère est ON DELETE RESTRICT : sans ce détachement, supprimer
    l'aller d'une mission canonique est refusé par PostgreSQL.
    """
    rows = CompanyManualBookingRequestOccurrence.query.filter_by(
        anchor_booking_id=int(booking_id)
    ).all()
    if not rows:
        return
    request_ids = {int(row.request_id) for row in rows}
    CompanyManualBookingRequestOccurrence.query.filter_by(
        anchor_booking_id=int(booking_id)
    ).delete(synchronize_session=False)
    for request_id in request_ids:
        remaining = CompanyManualBookingRequestOccurrence.query.filter_by(
            request_id=request_id
        ).count()
        if remaining == 0:
            CompanyManualBookingRequest.query.filter_by(id=request_id).delete(
                synchronize_session=False
            )


def company_mission_read_payload(booking: Any) -> dict[str, Any]:
    """Lecture : ancre canonique si elle existe, sinon synthèse sans écriture."""
    anchor = resolve_company_mission_anchor(booking)
    if anchor is not None:
        steps = [_step_payload(step) for step in _steps_for_anchor(int(anchor.id))]
        return {
            "mission_anchor_booking_id": int(anchor.id),
            "route_steps_source": "canonical",
            "route_steps": steps,
            "passenger_name": anchor.passenger_name,
            "external_reference": anchor.external_reference,
            "needs_assistance": bool(anchor.needs_assistance),
            "requester_name": anchor.requester_name,
            "requester_phone": anchor.requester_phone,
            "requester_service": anchor.requester_service,
            "pricing_mode": anchor.pricing_mode,
            "preferential_amount": (
                f"{Decimal(str(anchor.preferential_amount)):.2f}"
                if anchor.preferential_amount is not None
                else None
            ),
            "is_urgent": bool(anchor.is_urgent),
            "mission_type": anchor.mission_type,
            "billed_to_type": anchor.billed_to_type,
        }
    return {
        "mission_anchor_booking_id": int(booking.id) if getattr(booking, "id", None) else None,
        "route_steps_source": "legacy_synthesized",
        "route_steps": _synthesize_legacy_steps(booking),
        "passenger_name": None,
        "pricing_mode": None,
        "preferential_amount": None,
    }


def _opened_company_id(booking: Any) -> int | None:
    raw = getattr(booking, "company_id", None)
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def list_materialized_mission_segments(booking: Any) -> list[Any]:
    """Segments matérialisés du même tenant, sans filtre de statut.

    Le groupe et le parent ne servent jamais à sortir de l'entreprise du booking
    déjà autorisé par la route.
    """
    if booking is None or getattr(booking, "id", None) is None:
        return []
    company_id = _opened_company_id(booking)
    if company_id is None:
        return [booking]

    anchor = resolve_company_mission_anchor(booking)
    if anchor is not None and _opened_company_id(anchor) != company_id:
        anchor = None

    if anchor is not None:
        group_id = getattr(anchor, "route_group_id", None)
        if group_id:
            rows = (
                Booking.query.filter(
                    Booking.company_id == company_id,
                    Booking.route_group_id == group_id,
                )
                .order_by(Booking.route_sequence_number.asc(), Booking.id.asc())
                .all()
            )
            return rows or [booking]
        returns = (
            Booking.query.filter(
                Booking.company_id == company_id,
                Booking.parent_booking_id == int(anchor.id),
                Booking.is_return.is_(True),
            )
            .order_by(Booking.id.asc())
            .all()
        )
        return [anchor, *returns]

    base = booking
    parent_id = getattr(booking, "parent_booking_id", None)
    if getattr(booking, "is_return", False) and parent_id:
        parent = Booking.query.filter(
            Booking.id == int(parent_id),
            Booking.company_id == company_id,
        ).first()
        if parent is not None:
            base = parent
    child = (
        Booking.query.filter(
            Booking.company_id == company_id,
            Booking.parent_booking_id == int(base.id),
            Booking.is_return.is_(True),
        )
        .order_by(Booking.id.asc())
        .first()
    )
    if child is None:
        return [base]
    return [base, child]


def mission_segment_position(booking: Any) -> tuple[int, int]:
    """Index 1-based et nombre de segments matérialisés."""
    rows = list_materialized_mission_segments(booking)
    if not rows or getattr(booking, "id", None) is None:
        return 1, 1
    current = int(booking.id)
    ids = [int(row.id) for row in rows]
    if current not in ids:
        return 1, len(ids)
    return ids.index(current) + 1, len(ids)


def _mission_level_booking(booking: Any) -> Any:
    """Ancre canonique du même tenant, sinon booking historique du même tenant."""
    company_id = _opened_company_id(booking)
    anchor = resolve_company_mission_anchor(booking)
    if anchor is not None and _opened_company_id(anchor) == company_id:
        return anchor
    parent_id = getattr(booking, "parent_booking_id", None)
    if getattr(booking, "is_return", False) and parent_id and company_id is not None:
        parent = Booking.query.filter(
            Booking.id == int(parent_id),
            Booking.company_id == company_id,
        ).first()
        if parent is not None:
            return parent
    return booking


def company_mission_mobile_read(booking: Any) -> dict[str, Any]:
    """Parcours et champs mission. L'ancre étrangère n'est pas suivie."""
    company_id = _opened_company_id(booking)
    anchor = resolve_company_mission_anchor(booking)
    same_tenant_anchor = anchor is not None and _opened_company_id(anchor) == company_id
    if same_tenant_anchor:
        payload = company_mission_read_payload(booking)
    else:
        payload = {
            "mission_anchor_booking_id": int(booking.id) if getattr(booking, "id", None) else None,
            "route_steps_source": "legacy_synthesized",
            "route_steps": _synthesize_legacy_steps(booking, company_id=company_id),
            "passenger_name": None,
            "pricing_mode": None,
            "preferential_amount": None,
        }
    index, count = mission_segment_position(booking)
    payload["mission_segment_index"] = index
    payload["mission_segment_count"] = count
    mission = _mission_level_booking(booking)
    payload["needs_assistance"] = bool(getattr(mission, "needs_assistance", False))
    payload["requester_name"] = getattr(mission, "requester_name", None)
    payload["requester_phone"] = getattr(mission, "requester_phone", None)
    payload["mission_type"] = getattr(mission, "mission_type", None) or payload.get("mission_type")
    payload["delivery_description"] = getattr(mission, "delivery_description", None)
    payload["notes_medical"] = getattr(mission, "notes_medical", None)
    payload["wheelchair_client_has"] = bool(getattr(mission, "wheelchair_client_has", False))
    payload["wheelchair_need"] = bool(getattr(mission, "wheelchair_need", False))
    return payload


def apply_company_mission_mobile_read(summary: dict[str, Any], booking: Any) -> dict[str, Any]:
    """Ajoute la mission au summary sans remplacer le segment ouvert."""
    protected = {"id", "route", "status", "driver", "time", "client", "transfer", "flags"}
    read = company_mission_mobile_read(booking)
    for key, value in read.items():
        if key in protected:
            continue
        summary[key] = value
    return summary


def split_equal_cents(total: Decimal, count: int) -> list[Decimal]:
    cents = int((total * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    base, remainder = divmod(cents, count)
    amounts = [Decimal(base) / Decimal(100) for _ in range(count)]
    amounts[-1] = (Decimal(base + remainder) / Decimal(100)).quantize(Decimal("0.01"))
    return amounts


def _price_one_way_segment(
    *,
    company_id: int,
    client: Any,
    origin: dict[str, Any],
    destination: dict[str, Any],
    is_round_trip: bool,
) -> tuple[Decimal, bool]:
    """Cascade existante. Retourne (montant, split_total du modèle zone)."""
    from application.companies.reservations.create_manual_booking import (
        WEEKEND_START_INDEX,
        _amount_encodes_round_trip_total,
        _preferential_rate_to_booking_amount,
    )
    from models import PricingProfile
    from services.pricing.pricing_engine import compute_price

    if destination.get("location") in (None, ""):
        raise CompanyMissionError("Le tronçon n'a pas de destination.")
    per_leg = getattr(client, "preferential_rate", None)
    if per_leg and float(per_leg) > 0:
        amount = _preferential_rate_to_booking_amount(
            float(per_leg), is_round_trip=is_round_trip
        )
        return _money(amount), bool(is_round_trip)

    profile = (
        PricingProfile.query.filter_by(company_id=company_id, is_active=True)
        .order_by(PricingProfile.created_at.desc())
        .first()
    )
    version = None
    if profile is not None:
        version = profile.current_version or (
            sorted(profile.versions, key=lambda item: int(item.version), reverse=True)[0]
            if profile.versions
            else None
        )
    depart = origin.get("departure_at") or datetime.now()
    if version is None:
        return Decimal("0.00"), False
    context = {
        "is_weekend": depart.weekday() >= WEEKEND_START_INDEX,
        "is_round_trip": bool(is_round_trip),
        "pickup_local_time": depart.strftime("%H:%M"),
        "minutes_until_pickup": 9999,
        "distance_km": 0.0,
        "zones_count": 1,
    }
    amount, breakdown = compute_price({}, version, context)
    return _money(amount), _amount_encodes_round_trip_total(breakdown)


def resolve_company_mission_pricing(
    *,
    company_id: int,
    client: Any,
    steps: list[dict[str, Any]],
    pricing_mode: str,
    segment_amounts: list[dict[str, Any]] | None,
    preferential_amount: Any,
) -> list[dict[str, Any]]:
    """Même resolver pour le preview et la création."""
    pairs = list(range(len(steps) - 1))
    if pricing_mode == "manual":
        if not segment_amounts or len(segment_amounts) != len(pairs):
            raise CompanyMissionError("Chaque tronçon doit avoir un montant.")
        seen = {(int(row["from_position"]), int(row["to_position"])): row for row in segment_amounts}
        if len(seen) != len(pairs):
            raise CompanyMissionError("Les montants de tronçons sont incomplets.")
        priced = []
        for index in pairs:
            key = (index, index + 1)
            if key not in seen:
                raise CompanyMissionError("Une paire d'étapes n'a pas de montant.")
            priced.append(
                {
                    "from_position": index,
                    "to_position": index + 1,
                    "amount": f"{_money(seen[key]['amount']):.2f}",
                }
            )
        return priced

    if pricing_mode == "preferential":
        total = _money(preferential_amount)
        if total <= 0:
            raise CompanyMissionError("Le forfait doit être strictement positif.")
        parts = split_equal_cents(total, len(pairs))
        return [
            {
                "from_position": index,
                "to_position": index + 1,
                "amount": f"{part:.2f}",
            }
            for index, part in enumerate(parts)
        ]

    if pricing_mode != "automatic":
        raise CompanyMissionError("Mode de tarification inconnu.")

    if is_simple_round_trip(steps):
        amount, split_total = _price_one_way_segment(
            company_id=company_id,
            client=client,
            origin=steps[0],
            destination=steps[1],
            is_round_trip=True,
        )
        from application.companies.reservations.create_manual_booking import (
            _resolve_leg_amounts,
        )

        outbound, inbound, _, _ = _resolve_leg_amounts(
            float(amount),
            is_round_trip=True,
            price_total=float(amount) if split_total else None,
            preferential_per_leg=(
                float(client.preferential_rate)
                if getattr(client, "preferential_rate", None)
                and float(client.preferential_rate) > 0
                else None
            ),
            split_total=split_total,
        )
        return [
            {"from_position": 0, "to_position": 1, "amount": f"{_money(outbound):.2f}"},
            {"from_position": 1, "to_position": 2, "amount": f"{_money(inbound):.2f}"},
        ]

    priced = []
    for index in pairs:
        amount, _split = _price_one_way_segment(
            company_id=company_id,
            client=client,
            origin=steps[index],
            destination=steps[index + 1],
            is_round_trip=False,
        )
        priced.append(
            {
                "from_position": index,
                "to_position": index + 1,
                "amount": f"{amount:.2f}",
            }
        )
    return priced


def _shift_steps(steps: list[dict[str, Any]], delta: timedelta) -> list[dict[str, Any]]:
    shifted = []
    for step in steps:
        copy = dict(step)
        if copy["arrival_at"] is not None:
            copy["arrival_at"] = copy["arrival_at"] + delta
        if copy["departure_at"] is not None:
            copy["departure_at"] = copy["departure_at"] + delta
        shifted.append(copy)
    return shifted


def _within_recurrence_end(moment: datetime, end_raw: Any) -> bool:
    if not end_raw:
        return True
    try:
        end_date = parse_local_naive(str(end_raw))
    except Exception:
        return True
    if end_date is None:
        return True
    return moment <= end_date


def _recurrence_dates(first_departure: datetime, validated_data: dict[str, Any]) -> list[datetime]:
    """Même calendrier que la création historique (jour, semaine, jours choisis)."""
    dates = [first_departure]
    if not validated_data.get("is_recurring"):
        return dates
    recurrence_type = validated_data.get("recurrence_type") or "weekly"
    occurrences = int(validated_data.get("occurrences") or 1)
    recurrence_days = validated_data.get("recurrence_days") or []
    end_raw = validated_data.get("recurrence_end_date")
    if recurrence_type == "daily":
        for index in range(1, occurrences):
            next_date = first_departure + timedelta(days=index)
            if not _within_recurrence_end(next_date, end_raw):
                break
            dates.append(next_date)
    elif recurrence_type == "weekly":
        for index in range(1, occurrences):
            next_date = first_departure + timedelta(weeks=index)
            if not _within_recurrence_end(next_date, end_raw):
                break
            dates.append(next_date)
    elif recurrence_type == "custom" and recurrence_days:
        for target_weekday in recurrence_days:
            current_date = first_departure
            count = 0
            iteration = 0
            max_iterations = occurrences * 10
            while count < occurrences and iteration < max_iterations:
                iteration += 1
                if current_date.weekday() == int(target_weekday):
                    if not _within_recurrence_end(current_date, end_raw):
                        break
                    if current_date not in dates:
                        dates.append(current_date)
                    count += 1
                current_date += timedelta(days=1)
        dates = [item for item in dates if item is not None]
        dates.sort()
    return dates


def _occurrence_deltas(
    first_departure: datetime, validated_data: dict[str, Any]
) -> list[timedelta]:
    dates = _recurrence_dates(first_departure, validated_data)
    return [item - first_departure for item in dates]


def _resolve_billed_to(
    *,
    client: Any,
    validated_data: dict[str, Any],
    reference: datetime | None,
) -> tuple[str, int | None]:
    """Même payeur que la création historique : séjour clinique, sinon client."""
    explicit = validated_data.get("billed_to_type")
    if explicit:
        return str(explicit).lower(), validated_data.get("billed_to_company_id")

    from services.billing.client_stay_resolver import (
        find_active_stay_for_client,
        get_clinic_address_for_stay,
    )

    active_stay = None
    if reference is not None:
        active_stay = find_active_stay_for_client(
            client_id=int(validated_data["client_id"]),
            reference_date=reference,
        )
    bill_to_patient = bool(validated_data.get("bill_to_patient", False))
    if active_stay and not bill_to_patient:
        clinic_info = get_clinic_address_for_stay(active_stay) or {}
        clinic_id = clinic_info.get("clinic_id")
        if clinic_id:
            return "clinic", int(clinic_id)
    default = getattr(client, "default_billed_to_type", None) or "patient"
    return str(default).lower(), None


def _assert_pricing_mode_exclusive(mode: str, validated_data: dict[str, Any]) -> None:
    amounts = validated_data.get("segment_amounts")
    preferential = validated_data.get("preferential_amount")
    has_amounts = bool(amounts)
    has_preferential = preferential not in (None, "")
    if mode == "automatic" and (has_amounts or has_preferential):
        raise CompanyMissionError("Le mode automatique n'accepte pas de montant saisi.")
    if mode == "manual" and has_preferential:
        raise CompanyMissionError("Le mode manuel n'accepte pas de forfait.")
    if mode == "preferential" and has_amounts:
        raise CompanyMissionError("Le forfait n'accepte pas de montants par tronçon.")
    if mode not in ("automatic", "manual", "preferential"):
        raise CompanyMissionError("Mode de tarification inconnu.")


def _compact_address(value: Any) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _address_is_client_home(address: Any, home_address: Any) -> bool:
    """Vrai si l'arrêt reprend le domicile (rue + numéro), pas seulement la ville."""
    street = _compact_address(home_address)
    stop = _compact_address(address)
    if len(street) < 8 or not any(ch.isdigit() for ch in street) or not stop:
        return False
    return street in stop or stop in street


def _apply_home_access(booking: Booking, mission: dict[str, Any]) -> None:
    floor = mission.get("client_floor") or None
    door = mission.get("client_door_code") or None
    if not floor and not door:
        return
    home = mission.get("client_home_address")
    if _address_is_client_home(booking.pickup_location, home):
        if floor:
            booking.pickup_floor = floor
        if door:
            booking.pickup_door_code = door
    if _address_is_client_home(booking.dropoff_location, home):
        if floor:
            booking.dropoff_floor = floor
        if door:
            booking.dropoff_door_code = door


def _apply_mission_fields(booking: Booking, anchor_values: dict[str, Any]) -> None:
    booking.client_id = anchor_values["client_id"]
    booking.customer_name = anchor_values["customer_name"]
    booking.passenger_name = anchor_values["passenger_name"]
    booking.mission_type = anchor_values["mission_type"]
    booking.delivery_description = anchor_values["delivery_description"]
    booking.is_urgent = anchor_values["is_urgent"]
    booking.wheelchair_client_has = anchor_values["wheelchair_client_has"]
    booking.wheelchair_need = anchor_values["wheelchair_need"]
    booking.needs_assistance = anchor_values["needs_assistance"]
    booking.requester_name = anchor_values["requester_name"]
    booking.requester_phone = anchor_values["requester_phone"]
    booking.requester_service = anchor_values["requester_service"]
    booking.billed_to_type = anchor_values["billed_to_type"]
    booking.billed_to_company_id = anchor_values["billed_to_company_id"]
    booking.billed_to_contact = anchor_values["billed_to_contact"]
    booking.company_id = anchor_values["company_id"]
    booking.user_id = anchor_values["user_id"]
    booking.status = BookingStatus.ACCEPTED
    booking.booking_type = "manual"
    booking.created_via = BookingCreatedVia.DISPATCHER
    booking.billing_source = BillingSource.DEFAULT_CLIENT


def _materialize_occurrence(
    *,
    steps: list[dict[str, Any]],
    amounts: list[dict[str, Any]],
    mission: dict[str, Any],
    persist_steps: bool,
) -> tuple[Booking, list[Booking]]:
    import uuid

    pairs = [(steps[i], steps[i + 1]) for i in range(len(steps) - 1)]
    amount_by_pair = {
        (int(row["from_position"]), int(row["to_position"])): _money(row["amount"])
        for row in amounts
    }
    group_id = (
        str(uuid.uuid4()) if len(pairs) > 1 and not is_simple_round_trip(steps) else None
    )

    created: list[Booking] = []
    anchor: Booking | None = None
    for index, (origin, dest) in enumerate(pairs):
        booking = Booking()
        _apply_mission_fields(booking, mission)
        booking.pickup_location = origin["location"]
        booking.dropoff_location = dest["location"]
        booking.pickup_lat = origin.get("latitude")
        booking.pickup_lon = origin.get("longitude")
        booking.dropoff_lat = dest.get("latitude")
        booking.dropoff_lon = dest.get("longitude")
        is_return_leg = dest["kind"] == "return"
        departure_at = origin.get("departure_at")
        booking.is_return = is_return_leg
        if departure_at is None:
            booking.time_confirmed = False
            booking.scheduled_time = None
        else:
            booking.scheduled_time = departure_at
            booking.time_confirmed = True
        booking.amount = float(amount_by_pair[(index, index + 1)])
        booking.pickup_access_notes = origin.get("access_notes")
        booking.dropoff_access_notes = dest.get("access_notes")
        clinical = origin if is_return_leg else dest
        booking.medical_facility = clinical.get("establishment")
        booking.hospital_service = clinical.get("service")
        booking.doctor_name = clinical.get("doctor")
        _apply_home_access(booking, mission)
        booking.notes_medical = mission["notes_medical"]
        booking.is_round_trip = any(step["kind"] == "return" for step in steps) and index == 0
        if group_id:
            booking.route_group_id = group_id
            booking.route_sequence_number = index + 1
        if index == 0:
            booking.pricing_mode = mission["pricing_mode"]
            booking.preferential_amount = mission["preferential_amount"]
            booking.external_reference = mission["external_reference"]
        db.session.add(booking)
        db.session.flush()
        if anchor is None:
            anchor = booking
        elif booking.is_return:
            booking.parent_booking_id = anchor.id
            booking.is_round_trip = False
        created.append(booking)

    assert anchor is not None
    if persist_steps:
        for step in steps:
            db.session.add(
                CompanyBookingRouteStep(
                    anchor_booking_id=anchor.id,
                    position=step["position"],
                    kind=step["kind"],
                    location=step["location"],
                    latitude=step.get("latitude"),
                    longitude=step.get("longitude"),
                    arrival_at=step.get("arrival_at"),
                    departure_at=step.get("departure_at"),
                    time_confirmed=bool(step.get("departure_at") or step.get("arrival_at")),
                    destination_kind=step.get("destination_kind"),
                    establishment=step.get("establishment"),
                    service=step.get("service"),
                    doctor=step.get("doctor"),
                    access_notes=step.get("access_notes"),
                )
            )
    returns = [item for item in created if item.is_return]
    return anchor, returns


def _replay_request(request_row: CompanyManualBookingRequest) -> tuple[list[Booking], list[Booking]]:
    occurrences = (
        CompanyManualBookingRequestOccurrence.query.filter_by(request_id=request_row.id)
        .order_by(CompanyManualBookingRequestOccurrence.occurrence_index.asc())
        .all()
    )
    anchors: list[Booking] = []
    returns: list[Booking] = []
    for row in occurrences:
        anchor = db.session.get(Booking, row.anchor_booking_id)
        if anchor is None:
            continue
        anchors.append(anchor)
        children = Booking.query.filter_by(parent_booking_id=anchor.id, is_return=True).all()
        returns.extend(children)
        if anchor.route_group_id:
            siblings = (
                Booking.query.filter(
                    Booking.route_group_id == anchor.route_group_id,
                    Booking.id != anchor.id,
                    Booking.is_return.is_(False),
                )
                .order_by(Booking.route_sequence_number.asc(), Booking.id.asc())
                .all()
            )
            anchors.extend(siblings)
    return anchors, returns


def create_canonical_series(
    *,
    company_id: int,
    client: Any,
    user: Any,
    validated_data: dict[str, Any],
) -> tuple[list[Booking], list[Booking]]:
    """Crée toute la série dans la transaction courante. Ne commit pas."""
    from application.companies.reservations.create_manual_booking import (
        CreateManualBookingError,
    )

    try:
        return _create_canonical_series(
            company_id=company_id,
            client=client,
            user=user,
            validated_data=validated_data,
        )
    except CompanyMissionError as exc:
        db.session.rollback()
        raise CreateManualBookingError(
            exc.message, status_code=exc.status_code, error_code=exc.error_code
        ) from exc
    except Exception:
        db.session.rollback()
        raise


def _create_canonical_series(
    *,
    company_id: int,
    client: Any,
    user: Any,
    validated_data: dict[str, Any],
) -> tuple[list[Booking], list[Booking]]:
    present = {
        key
        for key in _LEGACY_STRUCTURAL
        if key in validated_data and validated_data.get(key) not in (None, "", False)
    }
    if present:
        raise CompanyMissionError(
            "Le parcours canonique n'accepte pas les champs de trajet historiques."
        )

    mission_type = (validated_data.get("mission_type") or "patient_transport").strip().lower()
    steps = normalize_route_steps(
        list(validated_data.get("route_steps") or []),
        mission_type=mission_type,
    )
    mode = str(validated_data.get("pricing_mode") or "").strip()
    _assert_pricing_mode_exclusive(mode, validated_data)

    idempotency_key = str(validated_data.get("idempotency_key") or "").strip()
    if not idempotency_key:
        raise CompanyMissionError("La clé d'idempotence est obligatoire.")

    digest = hash_canonical_request(validated_data)

    existing = CompanyManualBookingRequest.query.filter_by(
        company_id=company_id, idempotency_key=idempotency_key
    ).first()
    if existing is not None:
        if existing.payload_hash != digest:
            raise CompanyMissionError(
                "Cette clé d'idempotence a déjà servi pour une autre réservation.",
                status_code=409,
                error_code="IDEMPOTENCY_CONFLICT",
            )
        return _replay_request(existing)

    first = (getattr(user, "first_name", None) or "").strip()
    last = (getattr(user, "last_name", None) or "").strip()
    full_name = f"{first} {last}".strip()
    if bool(getattr(client, "is_institution", False)) and getattr(client, "institution_name", None):
        customer_name = client.institution_name
    else:
        customer_name = full_name or (getattr(user, "username", "") or "Client")

    billed_to_type, billed_to_company_id = _resolve_billed_to(
        client=client,
        validated_data=validated_data,
        reference=steps[0]["departure_at"],
    )
    raw_desc = (validated_data.get("delivery_description") or "").strip()
    mission = {
        "client_id": validated_data["client_id"],
        "customer_name": customer_name,
        "passenger_name": (validated_data.get("passenger_name") or None),
        "mission_type": mission_type,
        "delivery_description": " ".join(raw_desc.split()) if raw_desc else None,
        "is_urgent": bool(validated_data.get("is_urgent", False)),
        "wheelchair_client_has": bool(validated_data.get("wheelchair_client_has", False)),
        "wheelchair_need": bool(validated_data.get("wheelchair_need", False)),
        "needs_assistance": bool(validated_data.get("needs_assistance", False)),
        "requester_name": validated_data.get("requester_name"),
        "requester_phone": validated_data.get("requester_phone"),
        "requester_service": validated_data.get("requester_service"),
        "billed_to_type": billed_to_type,
        "billed_to_company_id": billed_to_company_id,
        "billed_to_contact": validated_data.get("billed_to_contact"),
        "company_id": company_id,
        "user_id": getattr(user, "id", None),
        "pricing_mode": mode,
        "preferential_amount": (
            _money(validated_data.get("preferential_amount"))
            if mode == "preferential"
            else None
        ),
        "external_reference": validated_data.get("external_reference"),
        "notes_medical": validated_data.get("notes_medical"),
        "client_home_address": getattr(client, "domicile_address", None),
        "client_floor": getattr(client, "floor", None),
        "client_door_code": getattr(client, "door_code", None),
    }

    first_departure = steps[0]["departure_at"]
    assert isinstance(first_departure, datetime)
    deltas = _occurrence_deltas(first_departure, validated_data)

    request_row = CompanyManualBookingRequest(
        company_id=company_id,
        idempotency_key=idempotency_key,
        payload_hash=digest,
    )
    db.session.add(request_row)
    try:
        db.session.flush()
    except IntegrityError:
        db.session.rollback()
        winner = None
        for _attempt in range(8):
            winner = CompanyManualBookingRequest.query.filter_by(
                company_id=company_id, idempotency_key=idempotency_key
            ).first()
            if winner is not None:
                break
            time.sleep(0.05)
        if winner is None or winner.payload_hash != digest:
            raise CompanyMissionError(
                "Cette clé d'idempotence a déjà servi pour une autre réservation.",
                status_code=409,
                error_code="IDEMPOTENCY_CONFLICT",
            ) from None
        return _replay_request(winner)

    all_outbounds: list[Booking] = []
    all_returns: list[Booking] = []
    for index, delta in enumerate(deltas):
        occurrence_steps = _shift_steps(steps, delta)
        priced = resolve_company_mission_pricing(
            company_id=company_id,
            client=client,
            steps=occurrence_steps,
            pricing_mode=mode,
            segment_amounts=validated_data.get("segment_amounts"),
            preferential_amount=validated_data.get("preferential_amount"),
        )
        anchor, returns = _materialize_occurrence(
            steps=occurrence_steps,
            amounts=priced,
            mission=mission,
            persist_steps=True,
        )
        db.session.add(
            CompanyManualBookingRequestOccurrence(
                request_id=request_row.id,
                occurrence_index=index,
                anchor_booking_id=anchor.id,
            )
        )
        all_outbounds.append(anchor)
        if anchor.route_group_id:
            siblings = (
                Booking.query.filter(
                    Booking.route_group_id == anchor.route_group_id,
                    Booking.id != anchor.id,
                    Booking.is_return.is_(False),
                )
                .order_by(Booking.route_sequence_number.asc(), Booking.id.asc())
                .all()
            )
            all_outbounds.extend(siblings)
        all_returns.extend(returns)
    return all_outbounds, all_returns


def commit_or_replay_idempotency(
    *,
    company_id: int,
    idempotency_key: str,
    payload_hash: str,
) -> tuple[list[Booking], list[Booking]] | None:
    """None si le commit a réussi. Sinon la série à rejouer, ou une erreur 409."""
    from application.companies.reservations.create_manual_booking import (
        CreateManualBookingError,
    )

    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        for _attempt in range(8):
            existing = CompanyManualBookingRequest.query.filter_by(
                company_id=company_id, idempotency_key=idempotency_key
            ).first()
            if existing is None:
                time.sleep(0.05)
                continue
            if existing.payload_hash != payload_hash:
                raise CreateManualBookingError(
                    "Cette clé d'idempotence a déjà servi pour une autre réservation.",
                    status_code=409,
                    error_code="IDEMPOTENCY_CONFLICT",
                ) from None
            return _replay_request(existing)
        raise CreateManualBookingError(
            "Création concurrente en conflit. Réessayez.",
            status_code=409,
            error_code="IDEMPOTENCY_CONFLICT",
        ) from None
    return None


def preview_company_mission_pricing(
    *,
    company_id: int,
    client: Any,
    validated_data: dict[str, Any],
) -> dict[str, Any]:
    """Aperçu sans écriture. Même calendrier et même resolver que la création."""
    mission_type = (validated_data.get("mission_type") or "patient_transport").strip().lower()
    steps = normalize_route_steps(
        list(validated_data.get("route_steps") or []),
        mission_type=mission_type,
    )
    mode = str(validated_data.get("pricing_mode") or "").strip()
    _assert_pricing_mode_exclusive(mode, validated_data)
    first_departure = steps[0]["departure_at"]
    assert isinstance(first_departure, datetime)
    occurrences: list[dict[str, Any]] = []
    for delta in _occurrence_deltas(first_departure, validated_data):
        shifted = _shift_steps(steps, delta)
        segments = resolve_company_mission_pricing(
            company_id=company_id,
            client=client,
            steps=shifted,
            pricing_mode=mode,
            segment_amounts=validated_data.get("segment_amounts"),
            preferential_amount=validated_data.get("preferential_amount"),
        )
        total = sum(Decimal(row["amount"]) for row in segments)
        depart = shifted[0]["departure_at"]
        assert isinstance(depart, datetime)
        occurrences.append(
            {
                "occurrence_date": depart.date().isoformat(),
                "segments": segments,
                "total": f"{total:.2f}",
            }
        )
    series = sum(Decimal(row["total"]) for row in occurrences)
    return {
        "pricing_mode": mode,
        "segments": occurrences[0]["segments"],
        "total": occurrences[0]["total"],
        "occurrences": occurrences,
        "series_total": f"{series:.2f}",
    }
