"""Barrière Google Geocoding : cache, plafond, débit, single-flight."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from services.geolocation import google_geocoding_gate as gate


def _ok_payload() -> dict:
    return {
        "status": "OK",
        "results": [{"geometry": {"location": {"lat": 46.2, "lng": 6.14}}}],
    }


def _mock_response(payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload or _ok_payload()
    return response


@pytest.fixture(autouse=True)
def _memory_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GEOCODING_DAILY_HARD_LIMIT", "200")
    monkeypatch.setenv("GEOCODING_PER_MINUTE_LIMIT", "20")
    gate.reset_geocoding_gate_for_tests()
    yield
    gate._force_memory = False
    gate._memory._values.clear()


def test_second_call_same_address_does_not_hit_google(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def _get(*_args, **_kwargs):
        calls["n"] += 1
        return _mock_response()

    monkeypatch.setattr(gate.requests, "get", _get)
    params = {"address": "Chemin des Courbes 9, 1247 Anières", "key": "secret"}
    first = gate.perform_google_geocode_http(params)
    second = gate.perform_google_geocode_http({**params, "key": "autre-cle"})
    assert first["status"] == "OK"
    assert second == first
    assert calls["n"] == 1
    assert gate.geocoding_daily_usage() == 1
    assert gate.geocoding_google_call_ratio() == 0.5


def test_cache_hit_lowers_google_call_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate.requests, "get", lambda *_a, **_k: _mock_response())
    params = {"address": "HUG", "key": "k"}
    gate.perform_google_geocode_http(params)
    gate.perform_google_geocode_http(params)
    assert gate.geocoding_daily_usage() == 1
    assert gate.geocoding_google_call_ratio() == 0.5


def test_autocomplete_enrichment_alerts_above_50_per_day(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GEOCODING_DAILY_HARD_LIMIT", "200")
    monkeypatch.setenv("GEOCODING_PER_MINUTE_LIMIT", "80")
    monkeypatch.setattr(gate.requests, "get", lambda *_a, **_k: _mock_response())
    with caplog.at_level("ERROR"), gate.geocoding_source("autocomplete_enrichment"):
        for index in range(51):
            gate.perform_google_geocode_http({"address": f"Rue {index}", "key": "k"})
    assert gate.geocoding_daily_usage(source="autocomplete_enrichment") == 51
    assert any(
        "autocomplete_enrichment a dépassé 50" in message for message in caplog.messages
    )


def test_daily_budget_blocks_before_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEOCODING_DAILY_HARD_LIMIT", "2")
    calls = {"n": 0}

    def _get(*_args, **_kwargs):
        calls["n"] += 1
        return _mock_response()

    monkeypatch.setattr(gate.requests, "get", _get)
    gate.perform_google_geocode_http({"address": "Rue A", "key": "k"})
    gate.perform_google_geocode_http({"address": "Rue B", "key": "k"})
    with pytest.raises(gate.GeocodingBudgetExceeded):
        gate.perform_google_geocode_http({"address": "Rue C", "key": "k"})
    assert calls["n"] == 2


def test_per_minute_limit_blocks_before_http(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEOCODING_PER_MINUTE_LIMIT", "2")
    calls = {"n": 0}

    def _get(*_args, **_kwargs):
        calls["n"] += 1
        return _mock_response()

    monkeypatch.setattr(gate.requests, "get", _get)
    gate.perform_google_geocode_http({"address": "Rue 1", "key": "k"})
    gate.perform_google_geocode_http({"address": "Rue 2", "key": "k"})
    with pytest.raises(gate.GeocodingRateLimited):
        gate.perform_google_geocode_http({"address": "Rue 3", "key": "k"})
    assert calls["n"] == 2


def test_usage_is_split_by_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate.requests, "get", lambda *_a, **_k: _mock_response())
    with gate.geocoding_source("pricing"):
        gate.perform_google_geocode_http({"address": "Tarif", "key": "k"})
    with gate.geocoding_source("not-a-real-source"):
        gate.perform_google_geocode_http({"address": "Inconnu", "key": "k"})
    assert gate.geocoding_daily_usage(source="pricing") == 1
    assert gate.geocoding_daily_usage(source="unspecified") == 1
    assert gate.geocoding_daily_usage(source="dispatch") == 0
    assert gate.geocoding_daily_usage() == 2


def test_singleflight_collapses_concurrent_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}
    release = threading.Event()

    def _get(*_args, **_kwargs):
        calls["n"] += 1
        release.wait(timeout=2)
        return _mock_response()

    monkeypatch.setattr(gate.requests, "get", _get)
    errors: list[BaseException] = []
    results: list[dict] = []

    def _worker() -> None:
        try:
            results.append(
                gate.perform_google_geocode_http(
                    {"address": "Rue des Bains 35, 1205 Genève", "key": "k"}
                )
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker) for _ in range(10)]
    for thread in threads:
        thread.start()
    threading.Event().wait(0.3)
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert calls["n"] == 1
    assert len(results) == 10
    assert gate.geocoding_daily_usage() == 1
    assert gate.geocoding_google_call_ratio() == 1.0
