from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services.geolocation import driver_location_http as dlh


def test_idempotent_store_then_get(monkeypatch: pytest.MonkeyPatch) -> None:
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

    payload = {"ok": True, "accept_status": "accepted_canonical"}
    dlh.store_idempotent_response(driver_id, key, payload)
    out = dlh.get_idempotent_response(driver_id, key)
    assert out == payload


def test_rate_limit_fail_open_without_redis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dlh, "redis_client", None)
    allowed, retry = dlh.check_http_driver_location_rate_limit(1)
    assert allowed is True
    assert retry is None
