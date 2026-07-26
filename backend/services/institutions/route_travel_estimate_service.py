"""Estimation durée trajet aller (1re étape) pour les demandes institution."""

from __future__ import annotations

import math
from typing import Any

from routes.geocode import match_alias
from services.geolocation.google_directions import (
    DirectionsLatLng,
    DirectionsRequest,
    fetch_directions,
)
from services.geolocation.google_places import geocode_address_google

ROAD_FACTOR = 1.4
FALLBACK_AVG_SPEED_KMH = 30
EARTH_RADIUS_KM = 6371


def _to_coord(value: Any) -> float | None:
    if value is None:
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n):
        return None
    if n == 0.0:
        return None
    return n


def _sorted_legs(request: Any) -> list[Any]:
    legs = list(getattr(request, "legs", None) or [])
    return sorted(legs, key=lambda leg: getattr(leg, "sequence_index", 0) or 0)


def resolve_outbound_route(request: Any) -> dict[str, Any]:
    """Trajet aller : coordonnées connues + adresses texte."""
    legs = _sorted_legs(request)
    if legs:
        first = legs[0]
        return {
            "pickup_lat": _to_coord(getattr(first, "pickup_lat", None))
            or _to_coord(getattr(request, "pickup_lat", None)),
            "pickup_lng": _to_coord(getattr(first, "pickup_lng", None))
            or _to_coord(getattr(request, "pickup_lng", None)),
            "dropoff_lat": _to_coord(getattr(first, "dropoff_lat", None)),
            "dropoff_lng": _to_coord(getattr(first, "dropoff_lng", None)),
            "pickup_address": str(
                getattr(first, "pickup_location", None)
                or getattr(request, "pickup_location", "")
                or ""
            ).strip(),
            "dropoff_address": str(
                getattr(first, "dropoff_location", None) or ""
            ).strip(),
        }
    return {
        "pickup_lat": _to_coord(getattr(request, "pickup_lat", None)),
        "pickup_lng": _to_coord(getattr(request, "pickup_lng", None)),
        "dropoff_lat": _to_coord(getattr(request, "dropoff_lat", None)),
        "dropoff_lng": _to_coord(getattr(request, "dropoff_lng", None)),
        "pickup_address": str(getattr(request, "pickup_location", "") or "").strip(),
        "dropoff_address": str(getattr(request, "dropoff_location", "") or "").strip(),
    }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _estimate_haversine_minutes(endpoints: dict[str, Any]) -> int | None:
    pickup_lat = endpoints.get("pickup_lat")
    pickup_lng = endpoints.get("pickup_lng")
    dropoff_lat = endpoints.get("dropoff_lat")
    dropoff_lng = endpoints.get("dropoff_lng")
    if None in (pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        return None
    straight_km = _haversine_km(pickup_lat, pickup_lng, dropoff_lat, dropoff_lng)
    if straight_km <= 0:
        return None
    road_km = straight_km * ROAD_FACTOR
    minutes = round((road_km / FALLBACK_AVG_SPEED_KMH) * 60)
    return max(5, minutes)


def _resolve_departure_unix(request: Any) -> int | None:
    from datetime import date, datetime

    next_confirmed = getattr(request, "next_confirmed_time", None)
    scheduled = getattr(request, "scheduled_time", None)
    mission_date = getattr(request, "mission_date", None)

    candidates: list[Any] = [next_confirmed, scheduled]
    if mission_date is not None and scheduled is None:
        candidates.append(mission_date)

    for raw in candidates:
        if raw is None:
            continue
        if isinstance(raw, datetime):
            return int(raw.timestamp())
        if isinstance(raw, date):
            return int(datetime.combine(raw, datetime.min.time()).timestamp())
        text = str(raw).strip()
        if not text:
            continue
        try:
            if "T" in text:
                dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            else:
                dt = datetime.fromisoformat(f"{text}T12:00:00")
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


def _geocode_address(address: str) -> tuple[float, float] | None:
    query = str(address or "").strip()
    if not query:
        return None

    alias = match_alias(query)
    if alias and alias.get("lat") is not None and alias.get("lon") is not None:
        return float(alias["lat"]), float(alias["lon"])

    try:
        result = geocode_address_google(query, country="CH")
    except Exception:
        result = None
    if not result:
        return None
    lat = _to_coord(result.get("lat"))
    lng = _to_coord(result.get("lon"))
    if lat is None or lng is None:
        return None
    return lat, lng


def ensure_outbound_coords(route: dict[str, Any]) -> dict[str, Any]:
    pickup_lat = route.get("pickup_lat")
    pickup_lng = route.get("pickup_lng")
    dropoff_lat = route.get("dropoff_lat")
    dropoff_lng = route.get("dropoff_lng")

    if pickup_lat is None or pickup_lng is None:
        coords = _geocode_address(route.get("pickup_address", ""))
        if coords:
            pickup_lat, pickup_lng = coords

    if dropoff_lat is None or dropoff_lng is None:
        coords = _geocode_address(route.get("dropoff_address", ""))
        if coords:
            dropoff_lat, dropoff_lng = coords

    return {
        "pickup_lat": pickup_lat,
        "pickup_lng": pickup_lng,
        "dropoff_lat": dropoff_lat,
        "dropoff_lng": dropoff_lng,
    }


def _fetch_google_minutes(
    endpoints: dict[str, Any], departure_unix: int | None = None
) -> tuple[int | None, str]:
    pickup_lat = endpoints.get("pickup_lat")
    pickup_lng = endpoints.get("pickup_lng")
    dropoff_lat = endpoints.get("dropoff_lat")
    dropoff_lng = endpoints.get("dropoff_lng")
    if None in (pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        return None, "missing_coords"

    attempts: list[int | None] = [departure_unix, None]
    seen: set[int | None] = set()
    for dep in attempts:
        if dep in seen:
            continue
        seen.add(dep)
        result = fetch_directions(
            DirectionsRequest(
                origin=DirectionsLatLng(pickup_lat, pickup_lng),
                destination=DirectionsLatLng(dropoff_lat, dropoff_lng),
                departure_time=dep if dep and dep > 0 else None,
            )
        )
        if result.status != "OK":
            continue
        traffic_sec = result.duration_in_traffic_seconds
        duration_sec = result.duration_seconds
        chosen = traffic_sec if traffic_sec and traffic_sec > 0 else duration_sec
        if chosen and chosen > 0:
            return max(1, round(chosen / 60)), "google_directions"

    return None, "directions_unavailable"


def estimate_outbound_travel_minutes(request: Any) -> dict[str, Any]:
    """Estime la durée aller en minutes (Google Directions, repli haversine)."""
    route = resolve_outbound_route(request)
    endpoints = ensure_outbound_coords(route)
    departure_unix = _resolve_departure_unix(request)

    minutes, source = _fetch_google_minutes(endpoints, departure_unix)
    if minutes is not None:
        return {"travel_minutes": minutes, "source": source}

    fallback = _estimate_haversine_minutes(endpoints)
    if fallback is not None:
        return {"travel_minutes": fallback, "source": "haversine"}

    return {"travel_minutes": None, "source": "unavailable"}
