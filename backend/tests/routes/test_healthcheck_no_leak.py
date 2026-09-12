"""Healthchecks : statut OK/error sans message technique."""

from __future__ import annotations

from services.monitoring import websocket_healthcheck as ws_health

SENTINEL = "SUPER_SECRET_INTERNAL_SENTINEL"


def test_check_websocket_health_redis_error_is_generic(monkeypatch):
    class _Redis:
        def ping(self):
            raise ConnectionError(f"redis://{SENTINEL}")

    monkeypatch.setattr(ws_health, "redis_client", _Redis())
    result = ws_health.check_websocket_health()
    assert result["redis_queue"] == "error"
    assert SENTINEL not in str(result)
