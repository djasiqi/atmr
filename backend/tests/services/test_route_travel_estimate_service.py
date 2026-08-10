"""Tests estimation trajet institution."""

from datetime import date, datetime
from types import SimpleNamespace

from services.institutions.route_travel_estimate_service import (
    ensure_outbound_coords,
    estimate_outbound_travel_minutes,
    resolve_outbound_route,
)


def test_resolve_outbound_route_uses_first_leg():
    request = SimpleNamespace(
        pickup_location="Institution",
        pickup_lat=None,
        pickup_lng=None,
        dropoff_location="Retour",
        dropoff_lat=None,
        dropoff_lng=None,
        legs=[
            SimpleNamespace(
                sequence_index=0,
                pickup_location="Chemin des Courbes 9, 1247, Anières",
                pickup_lat=None,
                pickup_lng=None,
                dropoff_location="HUG Genève",
                dropoff_lat=None,
                dropoff_lng=None,
            ),
            SimpleNamespace(
                sequence_index=1,
                pickup_location="HUG Genève",
                dropoff_location="Anières",
            ),
        ],
    )

    route = resolve_outbound_route(request)
    assert route["pickup_address"].startswith("Chemin des Courbes")
    assert "HUG" in route["dropoff_address"]


def test_estimate_anieres_hug_google_or_haversine(monkeypatch):
    request = SimpleNamespace(
        pickup_location="Chemin des Courbes 9, 1247, Anières",
        pickup_lat=46.2765,
        pickup_lng=6.2348,
        dropoff_location=(
            "Hôpitaux Universitaires de Genève (HUG), "
            "Rue Gabrielle-Perret-Gentil 4, 1205, Genève"
        ),
        dropoff_lat=46.1936,
        dropoff_lng=6.1489,
        mission_date=date(2026, 6, 16),
        scheduled_time=datetime(2026, 6, 16, 9, 30),
        next_confirmed_time=None,
        legs=[],
    )

    monkeypatch.setattr(
        "services.institutions.route_travel_estimate_service._geocode_address",
        lambda _address: None,
    )
    monkeypatch.setattr(
        "services.institutions.route_travel_estimate_service._fetch_google_minutes",
        lambda *_args, **_kwargs: (None, "directions_unavailable"),
    )

    payload = estimate_outbound_travel_minutes(request)
    assert payload["travel_minutes"] is not None
    assert payload["travel_minutes"] >= 15
    assert payload["source"] in {"google_directions", "haversine"}


def test_ensure_outbound_coords_uses_alias_for_hug():
    route = {
        "pickup_lat": 46.28,
        "pickup_lng": 6.23,
        "dropoff_lat": None,
        "dropoff_lng": None,
        "pickup_address": "Anières",
        "dropoff_address": "HUG Genève",
    }
    endpoints = ensure_outbound_coords(route)
    assert endpoints["dropoff_lat"] is not None
    assert endpoints["dropoff_lng"] is not None
