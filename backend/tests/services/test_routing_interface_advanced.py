from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from services import eta_service
from services.geolocation.routing_interface import set_routing_service
from services.unified_dispatch.data import build_time_matrix, calculate_eta
from services.unified_dispatch.settings import Settings


class _FakeRoutingService:
    def __init__(self) -> None:
        self.last_matrix_kwargs: dict[str, Any] | None = None
        self.last_eta_kwargs: dict[str, Any] | None = None

    def build_distance_matrix(self, coords, **kwargs):  # type: ignore[no-untyped-def]
        self.last_matrix_kwargs = {"coords": coords, **kwargs}
        n = len(coords)
        return [[0.0 for _ in range(n)] for _ in range(n)]

    def eta_seconds(self, origin, destination, **kwargs):  # type: ignore[no-untyped-def]
        self.last_eta_kwargs = {"origin": origin, "destination": destination, **kwargs}
        return 123

    def get_route(self, origin, destination, **kwargs):  # type: ignore[no-untyped-def]
        return {"duration": 123}

    def invalidate_cache(self, coords=None, zone_id=None):  # type: ignore[no-untyped-def]
        return None


def test_build_time_matrix_uses_routing_service_with_advanced_params(
    monkeypatch,
) -> None:
    fake = _FakeRoutingService()
    set_routing_service(fake)  # injection globale

    settings = Settings()
    settings.matrix.provider = "osrm"
    settings.matrix.osrm_url = "http://osrm-test:5000"
    settings.matrix.osrm_profile = "car"
    settings.matrix.osrm_timeout_sec = 7
    settings.matrix.osrm_max_retries = 3
    settings.matrix.osrm_max_sources_per_call = 55
    settings.matrix.osrm_rate_limit_per_sec = 9
    settings.matrix.osrm_retry_backoff_ms = 321

    drivers = [SimpleNamespace(current_lat=46.0, current_lon=6.0)]
    bookings = [
        SimpleNamespace(
            pickup_lat=46.1,
            pickup_lon=6.1,
            dropoff_lat=46.2,
            dropoff_lon=6.2,
        )
    ]

    matrix_min, coords_list, meta = build_time_matrix(
        bookings=bookings, drivers=drivers, settings=settings
    )

    assert fake.last_matrix_kwargs is not None
    assert fake.last_matrix_kwargs["base_url"] == "http://osrm-test:5000"
    assert fake.last_matrix_kwargs["profile"] == "car"
    assert fake.last_matrix_kwargs["timeout"] == 7
    assert fake.last_matrix_kwargs["max_sources_per_call"] == 55
    assert fake.last_matrix_kwargs["rate_limit_per_sec"] == 9
    assert fake.last_matrix_kwargs["max_retries"] == 3
    assert fake.last_matrix_kwargs["backoff_ms"] == 321

    assert coords_list
    assert matrix_min
    assert meta["provider"] == "osrm"


def test_calculate_eta_fallback_uses_routing_service_eta_seconds(monkeypatch) -> None:
    fake = _FakeRoutingService()
    set_routing_service(fake)

    # Forcer l'échec de EtaService pour prendre le chemin "legacy fallback"
    monkeypatch.setattr(
        eta_service, "get_eta_service", lambda: (_ for _ in ()).throw(Exception("boom"))
    )

    settings = Settings()
    settings.matrix.provider = "osrm"
    settings.matrix.osrm_url = "http://osrm-test:5000"
    settings.matrix.osrm_profile = "car"

    eta = calculate_eta(
        driver_position=(46.0, 6.0),
        destination=(46.1, 6.1),
        settings=settings,
        use_ml=False,
    )
    assert eta == 123
    assert fake.last_eta_kwargs is not None
    assert fake.last_eta_kwargs["base_url"] == "http://osrm-test:5000"

