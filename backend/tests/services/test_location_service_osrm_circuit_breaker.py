from __future__ import annotations

from datetime import UTC, datetime, timedelta

from services.geolocation.location import LocationService


def test_osrm_circuit_breaker_opens_after_threshold(monkeypatch):
    svc = LocationService(redis_client_instance=None)
    monkeypatch.setattr("services.geolocation.location.OSRM_CIRCUIT_BREAKER_THRESHOLD", 2)
    monkeypatch.setattr("services.geolocation.location.OSRM_CIRCUIT_BREAKER_COOLDOWN_SEC", 60)
    svc._register_osrm_failure("nearest", RuntimeError("timeout"))
    assert svc._is_osrm_circuit_open() is False
    svc._register_osrm_failure("nearest", RuntimeError("timeout"))
    assert svc._is_osrm_circuit_open() is True


def test_osrm_circuit_breaker_recovers_after_cooldown():
    svc = LocationService(redis_client_instance=None)
    svc._osrm_circuit_open_until = datetime.now(UTC) - timedelta(seconds=1)
    assert svc._is_osrm_circuit_open() is False
    assert svc._osrm_circuit_open_until is None


def test_osrm_success_resets_degraded_mode():
    svc = LocationService(redis_client_instance=None)
    svc._osrm_degraded_mode = True
    svc._osrm_failures = 7
    svc._register_osrm_success()
    assert svc._osrm_degraded_mode is False
    assert svc._osrm_failures == 0
