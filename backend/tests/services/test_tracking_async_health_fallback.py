"""Tests circuit breaker async tracking (heartbeat Redis + décision sync)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from services.tracking import async_circuit as ac


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    def get(self, key: str):
        return self.store.get(key)

    def setex(self, key: str, _ttl: int, value: str):
        self.store[key] = value if isinstance(value, str) else value.decode("utf-8")


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


def test_should_use_sync_when_redis_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ac, "redis_client", None)
    monkeypatch.setattr(ac, "HEALTH_GATE_ENABLED", True)
    assert ac.should_use_async_ingest() is False


def test_evaluate_opens_after_fail_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ac, "redis_client", fake)
    monkeypatch.setattr(ac, "HEALTH_GATE_ENABLED", True)
    monkeypatch.setattr(ac, "CIRCUIT_FAIL_THRESHOLD", 2)
    monkeypatch.setattr(ac, "_consecutive_fail", 0)
    monkeypatch.setattr(ac, "_consecutive_ok", 0)
    monkeypatch.setattr(ac, "_last_eval_at", 0.0)
    # Pas de heartbeat → unhealthy
    for _ in range(2):
        monkeypatch.setattr(ac, "_last_eval_at", 0.0)
        payload = ac.evaluate_and_store_circuit(force=True)
    assert payload["state"] == "open"
    assert payload["reason"] == "heartbeat_absent"


def test_write_heartbeat_even_without_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ac, "redis_client", fake)
    ac.write_consumer_heartbeat()
    raw = fake.get(ac.HEARTBEAT_KEY)
    assert raw is not None
    data = json.loads(raw)
    assert "last_poll_at" in data
