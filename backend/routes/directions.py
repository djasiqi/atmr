"""REST proxy ``/directions`` pour la flotte mobile/web.

Le client envoie les coordonnées (origin, destination, waypoints optionnels) et reçoit
la polyline encodée Google. La clé Google reste côté serveur, les renders fréquents sont
absorbés par un cache Redis (clé déterministe sur le couple origin/destination/waypoints).
"""

from __future__ import annotations

import logging
from typing import Any

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource

from services.geolocation import google_directions
from services.geolocation.google_directions import (
    DirectionsLatLng,
    DirectionsRequest,
)
from services.monitoring.route_timing import route_duration_span

logger = logging.getLogger(__name__)

directions_ns = Namespace(
    "directions",
    description="Proxy serveur Google Directions (cache Redis)",
)

_MAX_WAYPOINTS = 10


def _coerce_lat_lng(payload: Any) -> DirectionsLatLng | None:
    if not isinstance(payload, dict):
        return None
    lat = payload.get("latitude", payload.get("lat"))
    lng = payload.get("longitude", payload.get("lng"))
    try:
        latitude = float(lat)  # type: ignore[arg-type]
        longitude = float(lng)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= latitude <= 90.0):
        return None
    if not (-180.0 <= longitude <= 180.0):
        return None
    return DirectionsLatLng(latitude=latitude, longitude=longitude)


def _coerce_waypoints(payload: Any) -> tuple[DirectionsLatLng, ...]:
    if not isinstance(payload, list):
        return ()
    waypoints: list[DirectionsLatLng] = []
    for item in payload[:_MAX_WAYPOINTS]:
        coerced = _coerce_lat_lng(item)
        if coerced is not None:
            waypoints.append(coerced)
    return tuple(waypoints)


@directions_ns.route("")
@directions_ns.route("/")
class DirectionsProxy(Resource):
    """``POST /directions`` — renvoie la polyline encodée pour origin/destination."""

    @jwt_required()
    def post(self):
        body = request.get_json(silent=True) or {}
        origin = _coerce_lat_lng(body.get("origin"))
        destination = _coerce_lat_lng(body.get("destination"))
        if origin is None or destination is None:
            return {
                "error": "origin et destination requis (lat/lng valides)",
            }, 400

        mode_raw = body.get("mode")
        mode = (
            mode_raw
            if isinstance(mode_raw, str) and mode_raw.strip()
            else "driving"
        )
        region_raw = body.get("region")
        departure_raw = body.get("departure_time")
        departure_time: int | None = None
        if departure_raw is not None:
            try:
                departure_time = int(departure_raw)
            except (TypeError, ValueError):
                departure_time = None

        directions_request = DirectionsRequest(
            origin=origin,
            destination=destination,
            waypoints=_coerce_waypoints(body.get("waypoints")),
            mode=mode.lower(),
            region=(
                region_raw.lower()
                if isinstance(region_raw, str) and region_raw.strip()
                else "ch"
            ),
            departure_time=departure_time,
        )

        with route_duration_span(
            "directions.proxy",
            mode=directions_request.mode,
            waypoints=len(directions_request.waypoints),
        ):
            result = google_directions.fetch_directions(directions_request)

        payload: dict[str, Any] = {
            "status": result.status,
            "overview_polyline": result.overview_polyline,
            "duration_seconds": result.duration_seconds,
            "distance_meters": result.distance_meters,
            "duration_in_traffic_seconds": result.duration_in_traffic_seconds,
            "cached": result.cached,
            "source": "google_directions_v1",
        }
        if result.error_message:
            payload["error_message"] = result.error_message
        if result.http_status is not None:
            payload["http_status"] = result.http_status
        return payload, 200
