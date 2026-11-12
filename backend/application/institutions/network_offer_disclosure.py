"""Divulgation d'une offre réseau avant acceptation.

Les zones viennent du référentiel local déjà utilisé pour l'éligibilité.
Une zone non reconnue reste non confirmée : aucune rue n'est découpée
pour fabriquer un secteur. Aucun géocodage externe n'est appelé.
"""

from __future__ import annotations

import math
from typing import Any

from application.institutions.network_eligibility import (
    DISPATCH_NETWORK,
    _local_unit,
)
from models.enums import GeoUnitType, OfferStatus
from services.geo.geo_resolver import geo_chain

_EARTH_RADIUS_KM = 6371.0
_ROAD_FACTOR = 1.4
_AVG_SPEED_KMH = 30.0


def precise_itinerary_allowed(offer: Any) -> bool:
    """Vrai pour une offre partenaire, ou une offre réseau déjà acceptée."""
    request = getattr(offer, "transport_request", None)
    target = str(getattr(request, "dispatch_target", "") or "").strip().lower()
    if target != DISPATCH_NETWORK:
        return True
    status = getattr(offer, "status", None)
    status_value = getattr(status, "value", status)
    return str(status_value or "") == OfferStatus.ACCEPTED.value


def confirmed_zone_label(location: str | None) -> str | None:
    """Nom de commune ou de canton du référentiel, sinon rien."""
    unit = _local_unit(location)
    if unit is None:
        return None
    name = str(getattr(unit, "name", "") or "").strip()
    if not name:
        return None
    canton_name = ""
    for item in geo_chain(unit):
        if getattr(item, "type", None) == GeoUnitType.CANTON:
            canton_name = str(getattr(item, "name", "") or "").strip()
            break
    if canton_name and canton_name.casefold() != name.casefold():
        return f"{name}, {canton_name}"
    return name


BIRD_FLIGHT_NOTICE = (
    "Distance indicative à vol d'oiseau. Ce n'est pas une distance routière "
    "et ce n'est pas un kilométrage facturable."
)


def indicative_route_hint(request: Any) -> dict[str, Any]:
    """Distance et durée à vol d'oiseau, sans coordonnées dans la réponse.

    Utilise seulement les coordonnées déjà enregistrées. N'appelle ni
    Google ni un géocodeur.
    """
    pickup_lat, pickup_lng, dropoff_lat, dropoff_lng = _stored_endpoints(request)
    if None in (pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        return {
            "distance_km": None,
            "travel_minutes": None,
            "source": "unavailable",
        }
    km = _haversine_km(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
    if km <= 0:
        return {
            "distance_km": None,
            "travel_minutes": None,
            "source": "unavailable",
        }
    rounded_km = max(1, round(km))
    minutes = max(5, round((km * _ROAD_FACTOR / _AVG_SPEED_KMH) * 60))
    return {
        "distance_km": rounded_km,
        "travel_minutes": minutes,
        "source": "vol_oiseau",
        "distance_notice": BIRD_FLIGHT_NOTICE,
    }


def apply_pending_network_disclosure(view: dict[str, Any], request: Any) -> None:
    """Retire l'itinéraire exact et les données cliniques d'une offre non acceptée."""
    for key in (
        "external_reference",
        "patient_name",
        "patient",
        "delivery_description",
        "notes",
        "contact_on_site",
        "pickup_location",
        "dropoff_location",
        "pickup_lat",
        "pickup_lng",
        "dropoff_lat",
        "dropoff_lng",
    ):
        view[key] = None

    view["pickup_zone"] = confirmed_zone_label(
        getattr(request, "pickup_location", None)
    )
    view["dropoff_zone"] = confirmed_zone_label(
        getattr(request, "dropoff_location", None)
    )
    view["pickup_zone_confirmed"] = view["pickup_zone"] is not None
    view["dropoff_zone_confirmed"] = view["dropoff_zone"] is not None
    view["route_hint"] = indicative_route_hint(request)
    view["disclosure"] = "network_pending"
    view["legs"] = [
        _redacted_leg(leg) for leg in _sorted_legs(getattr(request, "legs", None))
    ]


def travel_estimate_for_offer(offer: Any) -> dict[str, Any]:
    """Estimation visible par l'entreprise, sans coordonnées."""
    if not precise_itinerary_allowed(offer):
        hint = indicative_route_hint(getattr(offer, "transport_request", None))
        payload = {
            "travel_minutes": hint["travel_minutes"],
            "distance_km": hint["distance_km"],
            "source": hint["source"],
        }
        if hint.get("distance_notice"):
            payload["distance_notice"] = hint["distance_notice"]
        return payload

    from services.institutions.route_travel_estimate_service import (
        estimate_outbound_travel_minutes,
    )

    payload = estimate_outbound_travel_minutes(
        getattr(offer, "transport_request", None)
    )
    return {
        "travel_minutes": payload.get("travel_minutes"),
        "source": payload.get("source"),
    }


def _redacted_leg(leg: Any) -> dict[str, Any]:
    from models.transport_request_leg import _iso_scheduled

    pickup_zone = confirmed_zone_label(getattr(leg, "pickup_location", None))
    dropoff_zone = confirmed_zone_label(getattr(leg, "dropoff_location", None))
    return {
        "sequence_index": getattr(leg, "sequence_index", None),
        "route_sequence_number": getattr(leg, "route_sequence_number", None),
        "scheduled_time": _iso_scheduled(getattr(leg, "scheduled_time", None)),
        "time_confirmed": bool(getattr(leg, "time_confirmed", False)),
        "is_return_stop": bool(getattr(leg, "is_return_stop", False)),
        "pickup_location": None,
        "dropoff_location": None,
        "pickup_lat": None,
        "pickup_lng": None,
        "dropoff_lat": None,
        "dropoff_lng": None,
        "dropoff_establishment": None,
        "dropoff_service": None,
        "dropoff_doctor": None,
        "pickup_zone": pickup_zone,
        "dropoff_zone": dropoff_zone,
        "pickup_zone_confirmed": pickup_zone is not None,
        "dropoff_zone_confirmed": dropoff_zone is not None,
    }


def _sorted_legs(legs: Any) -> list[Any]:
    return sorted(legs or [], key=lambda leg: getattr(leg, "sequence_index", 0) or 0)


def _stored_endpoints(
    request: Any,
) -> tuple[float | None, float | None, float | None, float | None]:
    legs = _sorted_legs(getattr(request, "legs", None))
    first = legs[0] if legs else None
    pickup_lat = _coord(getattr(first, "pickup_lat", None) if first else None)
    pickup_lng = _coord(getattr(first, "pickup_lng", None) if first else None)
    dropoff_lat = _coord(getattr(first, "dropoff_lat", None) if first else None)
    dropoff_lng = _coord(getattr(first, "dropoff_lng", None) if first else None)
    if pickup_lat is None:
        pickup_lat = _coord(getattr(request, "pickup_lat", None))
    if pickup_lng is None:
        pickup_lng = _coord(getattr(request, "pickup_lng", None))
    if dropoff_lat is None:
        dropoff_lat = _coord(getattr(request, "dropoff_lat", None))
    if dropoff_lng is None:
        dropoff_lng = _coord(getattr(request, "dropoff_lng", None))
    return pickup_lat, pickup_lng, dropoff_lat, dropoff_lng


def _coord(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number == 0.0:
        return None
    return number


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return _EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
