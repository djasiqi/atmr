"""Intégration OSRM indisponible / intermittent — fallback coords raw + circuit breaker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests
from requests import Timeout

from services.geolocation.location import LocationService


def test_osrm_down_nearest_returns_none_and_records_timeout(monkeypatch):
    svc = LocationService(redis_client_instance=None)
    calls: list[str] = []

    def _observe(*, operation: str, result: str, duration_sec: float) -> None:
        calls.append(f"{operation}:{result}")

    monkeypatch.setattr(
        "services.geolocation.location.observe_osrm_request",
        _observe,
    )

    def _timeout_get(*_args, **_kwargs):
        raise Timeout("osrm down")

    monkeypatch.setattr("services.geolocation.location.requests.get", _timeout_get)

    result = svc._snap_to_road(6.14, 46.20)
    assert result is None
    assert any("nearest:timeout" in c for c in calls)


def test_osrm_intermittent_opens_circuit_breaker(monkeypatch):
    svc = LocationService(redis_client_instance=None)
    monkeypatch.setattr(
        "services.geolocation.location.OSRM_CIRCUIT_BREAKER_THRESHOLD", 3
    )
    monkeypatch.setattr(
        "services.geolocation.location.observe_osrm_request",
        lambda **_kwargs: None,
    )

    def _timeout_get(*_args, **_kwargs):
        raise Timeout("slow")

    monkeypatch.setattr("services.geolocation.location.requests.get", _timeout_get)

    svc._snap_to_road(6.14, 46.20)
    svc._snap_to_road(6.15, 46.21)
    svc._snap_to_road(6.16, 46.22)
    assert svc._is_osrm_circuit_open() is True

    with patch("services.geolocation.location.requests.get") as mock_get:
        svc._snap_to_road(6.17, 46.23)
        mock_get.assert_not_called()


def test_osrm_snap_timeout_disabled_uses_legacy_timeout(monkeypatch):
    svc = LocationService(redis_client_instance=None)
    monkeypatch.setattr(
        "services.geolocation.location.OSRM_SNAP_TIMEOUT_ENABLED", False
    )
    captured: dict[str, float] = {}

    def _get(_url, params=None, timeout=None):
        captured["timeout"] = timeout
        r = MagicMock()
        r.ok = True
        r.json.return_value = {"waypoints": [{"location": [6.14, 46.2]}]}
        return r

    monkeypatch.setattr(
        "services.geolocation.location.observe_osrm_request",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr("services.geolocation.location.requests.get", _get)

    svc._snap_to_road(6.14, 46.20)
    assert captured.get("timeout") == 2.0
