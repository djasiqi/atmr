from __future__ import annotations

from services.monitoring.websocket_rate_limiter import WebSocketRateLimiter


class _FakeRedis:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.allowed_count = 0

    def eval(self, _script, _nkeys, key, now_ts, window_seconds, limit, member):
        self.calls.append((key, now_ts, window_seconds, limit, member))
        if self.allowed_count < int(limit):
            self.allowed_count += 1
            return [1, 0, self.allowed_count]
        return [0, 1, self.allowed_count]


def test_rate_limiter_uses_atomic_lua_eval(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr("services.monitoring.websocket_rate_limiter.redis_client", fake)
    limiter = WebSocketRateLimiter()
    allowed_1, retry_1 = limiter.check_rate_limit("driver_location", driver_id=10)
    allowed_2, retry_2 = limiter.check_rate_limit("driver_location", driver_id=10)
    assert allowed_1 is True
    assert retry_1 is None
    assert allowed_2 is False
    assert retry_2 == 1
    assert len(fake.calls) == 2


def test_rate_limiter_fallback_memory_when_redis_unavailable(monkeypatch):
    monkeypatch.setattr("services.monitoring.websocket_rate_limiter.redis_client", None)
    limiter = WebSocketRateLimiter()
    limiter.use_redis = False
    allowed_1, _ = limiter.check_rate_limit("driver_location", driver_id=22)
    allowed_2, retry_2 = limiter.check_rate_limit("driver_location", driver_id=22)
    assert allowed_1 is True
    assert allowed_2 is False
    assert retry_2 is not None
