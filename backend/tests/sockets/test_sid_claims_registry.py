"""Tests sid_claims_registry (Redis + fallback local)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.realtime import sid_claims_registry as reg


@pytest.fixture(autouse=True)
def _clear_local_cache():
    reg._LOCAL_SID_CLAIMS.clear()
    yield
    reg._LOCAL_SID_CLAIMS.clear()


def test_set_get_delete_local_fallback():
    with patch.object(reg, "redis_client", None):
        reg.set_sid_claims("sid-1", {"user_public_id": "u1", "user_id": 42})
        assert reg.get_sid_claims("sid-1")["user_id"] == 42
        removed = reg.delete_sid_claims("sid-1")
        assert removed is not None
        assert reg.get_sid_claims("sid-1") == {}


def test_redis_roundtrip():
    store: dict[str, str] = {}
    mock_redis = MagicMock()

    def _setex(key, ttl, value):
        store[key] = value

    def _get(key):
        return store.get(key)

    def _delete(key):
        store.pop(key, None)

    mock_redis.setex = _setex
    mock_redis.get = _get
    mock_redis.delete = _delete

    with patch.object(reg, "redis_client", mock_redis):
        reg.set_sid_claims("sid-redis", {"user_public_id": "abc", "role": "driver"})
        claims = reg.get_sid_claims("sid-redis")
        assert claims["user_public_id"] == "abc"
        reg.delete_sid_claims("sid-redis")
        assert reg.get_sid_claims("sid-redis") == {}
