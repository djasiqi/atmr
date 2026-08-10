"""Tests limiteur HTTP GPS dual-fenêtre (Lua atomique + fallback mémoire)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services.geolocation import driver_location_http as dlh


def test_idempotent_store_only_persisted_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    storage: dict[str, bytes] = {}

    def fake_get(k: str | bytes) -> bytes | None:
        ks = k.decode("utf-8") if isinstance(k, bytes) else k
        return storage.get(ks)

    def fake_setex(k: str | bytes, _ttl: int, v: str | bytes) -> None:
        ks = k.decode("utf-8") if isinstance(k, bytes) else k
        storage[ks] = v if isinstance(v, bytes) else v.encode("utf-8")

    fake = MagicMock()
    fake.get = fake_get
    fake.setex = fake_setex
    monkeypatch.setattr(dlh, "redis_client", fake)

    driver_id = 7
    key = "idem-key-abc"
    assert dlh.get_idempotent_response(driver_id, key) is None

    # 202 queued_async ne doit pas être caché
    dlh.store_idempotent_response(
        driver_id,
        key,
        {"ok": True, "queued": True, "durability": "queued_async"},
    )
    assert dlh.get_idempotent_response(driver_id, key) is None

    payload = {
        "ok": True,
        "ack_status": "persisted",
        "durability": "persisted_sync",
        "location_event_id": "trk_1",
    }
    dlh.store_idempotent_response(driver_id, key, payload)
    out = dlh.get_idempotent_response(driver_id, key)
    assert out == payload


def test_rate_limit_memory_fallback_when_redis_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dlh, "redis_client", None)
    monkeypatch.setattr(dlh, "_MEMORY_FALLBACK_LIMIT", 2)
    monkeypatch.setattr(dlh, "_memory_hits", {})
    a1, _r1, reason1 = dlh.check_http_driver_location_rate_limit(42)
    a2, _r2, reason2 = dlh.check_http_driver_location_rate_limit(42)
    a3, r3, reason3 = dlh.check_http_driver_location_rate_limit(42)
    assert a1 is True
    assert reason1 is None
    assert a2 is True
    assert reason2 is None
    assert a3 is False
    assert r3 is not None
    assert r3 >= 1
    assert reason3 == "memory_fallback"


def test_dual_window_lua_rejects_at_limit_without_zadd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """count >= limit → reject ; membre unique passé à eval."""
    calls: list[tuple] = []

    class FakeRedis:
        def eval(self, script, nkeys, *args):
            calls.append((nkeys, args))
            # Simule fenêtre short saturée
            return [0, 3, 30, 10, "short_window"]

    monkeypatch.setattr(dlh, "redis_client", FakeRedis())
    allowed, retry, reason = dlh.check_http_driver_location_rate_limit(9)
    assert allowed is False
    assert retry == 3
    assert reason == "short_window"
    assert len(calls) == 1
    nkeys, args = calls[0]
    assert nkeys == 2
    # member unique contient ms:uuid
    member = args[7]
    assert ":" in str(member)


def test_dual_window_lua_accepts_and_adds(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeRedis:
        def eval(self, script, nkeys, *args):
            return [1, 0, 1, 1, "ok"]

    monkeypatch.setattr(dlh, "redis_client", FakeRedis())
    allowed, retry, reason = dlh.check_http_driver_location_rate_limit(9)
    assert allowed is True
    assert retry is None
    assert reason is None


def test_lua_script_uses_gte_semantics() -> None:
    """Le script Lua doit contenir >= (pas >) pour éviter la 31e requête."""
    assert "short_count >= short_limit" in dlh._DUAL_WINDOW_LUA
    assert "long_count >= long_limit" in dlh._DUAL_WINDOW_LUA
