"""Tests circuit breaker async tracking (état Redis partagé, route GET only)."""

from __future__ import annotations

import json

import pytest

from services.tracking import async_circuit as ac


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str):
        return self.store.get(key)

    def setex(self, key: str, _ttl: int, value: str):
        self.store[key] = value if isinstance(value, str) else value.decode("utf-8")

    def delete(self, key: str):
        self.store.pop(key, None)


def test_should_use_async_when_circuit_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ac, "redis_client", fake)
    monkeypatch.setattr(ac, "HEALTH_GATE_ENABLED", True)
    monkeypatch.setenv("TRACKING_INGEST_ASYNC_ENABLED", "true")
    fake.setex(
        ac.CIRCUIT_KEY,
        60,
        json.dumps({"state": "closed", "evaluated_at": ac._utcnow_iso()}),
    )
    ac.write_consumer_heartbeat(lag=0)
    assert ac.should_use_async_ingest() is True


def test_should_use_sync_when_circuit_open(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ac, "redis_client", fake)
    monkeypatch.setattr(ac, "HEALTH_GATE_ENABLED", True)
    monkeypatch.setenv("TRACKING_INGEST_ASYNC_ENABLED", "true")
    fake.setex(
        ac.CIRCUIT_KEY,
        60,
        json.dumps({"state": "open", "evaluated_at": ac._utcnow_iso()}),
    )
    assert ac.should_use_async_ingest() is False


def test_should_use_sync_when_redis_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ac, "redis_client", None)
    monkeypatch.setattr(ac, "HEALTH_GATE_ENABLED", True)
    assert ac.should_use_async_ingest() is False


def test_get_circuit_state_is_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """La route ne doit pas évaluer / écrire le circuit."""
    fake = _FakeRedis()
    monkeypatch.setattr(ac, "redis_client", fake)
    called = {"n": 0}

    def boom(**_kwargs):
        called["n"] += 1
        raise AssertionError("evaluate must not be called from get_circuit_state")

    monkeypatch.setattr(ac, "evaluate_and_store_circuit", boom)
    assert ac.get_circuit_state()["state"] == "open"
    assert ac.get_circuit_state()["reason"] == "circuit_absent"
    assert called["n"] == 0


def test_evaluate_persists_counters_in_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ac, "redis_client", fake)
    monkeypatch.setattr(ac, "HEALTH_GATE_ENABLED", True)
    monkeypatch.setattr(ac, "CIRCUIT_FAIL_THRESHOLD", 2)
    # Pas de heartbeat → unhealthy
    p1 = ac.evaluate_and_store_circuit(force=True)
    assert p1["consecutive_fail"] == 1
    p2 = ac.evaluate_and_store_circuit(force=True)
    assert p2["state"] == "open"
    assert p2["consecutive_fail"] == 2
    stored = json.loads(fake.get(ac.CIRCUIT_KEY))
    assert stored["consecutive_fail"] == 2


def test_open_circuit_immediate_deletes_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ac, "redis_client", fake)
    ac.write_consumer_heartbeat(lag=12)
    assert fake.get(ac.HEARTBEAT_KEY) is not None
    payload = ac.open_circuit_immediate(reason="consumer_down")
    assert payload["state"] == "open"
    assert payload["reason"] == "consumer_down"
    assert fake.get(ac.HEARTBEAT_KEY) is None


def test_heartbeat_carries_lag(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ac, "redis_client", fake)
    ac.write_consumer_heartbeat(lag=777)
    data = json.loads(fake.get(ac.HEARTBEAT_KEY))
    assert data["lag"] == 777


def test_open_state_respects_open_min_before_half_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ac, "redis_client", fake)
    monkeypatch.setattr(ac, "HEALTH_GATE_ENABLED", True)
    monkeypatch.setattr(ac, "CIRCUIT_OPEN_MIN_SEC", 30)
    monkeypatch.setattr(ac, "CIRCUIT_OK_THRESHOLD", 3)
    # Circuit open récent + heartbeat sain → doit rester open (pas closed)
    fake.setex(
        ac.CIRCUIT_KEY,
        60,
        json.dumps(
            {
                "state": "open",
                "opened_at": ac._utcnow_iso(),
                "evaluated_at": ac._utcnow_iso(),
                "consecutive_fail": 3,
                "consecutive_ok": 0,
            }
        ),
    )
    ac.write_consumer_heartbeat(lag=0)
    # 3 évaluations saines en < OPEN_MIN
    for _ in range(3):
        p = ac.evaluate_and_store_circuit(force=True)
    assert p["state"] == "open"
    assert p["consecutive_ok"] >= 1


def test_should_use_async_false_when_heartbeat_stale_even_if_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ac, "redis_client", fake)
    monkeypatch.setattr(ac, "HEALTH_GATE_ENABLED", True)
    monkeypatch.setenv("TRACKING_INGEST_ASYNC_ENABLED", "true")
    fake.setex(
        ac.CIRCUIT_KEY,
        60,
        json.dumps({"state": "closed", "evaluated_at": ac._utcnow_iso()}),
    )
    # Pas de heartbeat → sync (protecte kill -9)
    assert ac.should_use_async_ingest() is False
    ac.write_consumer_heartbeat(lag=0)
    assert ac.should_use_async_ingest() is True
