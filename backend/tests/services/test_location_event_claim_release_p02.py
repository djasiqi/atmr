"""P0.2 — claim Redis release + pas de faux persisted_sync sur duplicate."""

from __future__ import annotations

from services.geolocation import driver_location_dedup as d


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttl_sec: dict[str, int] = {}

    def set(self, key, value, nx=False, ex=None):  # type: ignore[no-untyped-def]
        if nx and key in self.store:
            return False
        self.store[key] = value
        if ex is not None:
            self.ttl_sec[key] = int(ex)
        return True

    def delete(self, key):  # type: ignore[no-untyped-def]
        existed = key in self.store
        self.store.pop(key, None)
        self.ttl_sec.pop(key, None)
        return 1 if existed else 0

    def ttl(self, key):  # type: ignore[no-untyped-def]
        if key not in self.store:
            return -2
        return int(self.ttl_sec.get(key, -1))


def test_claim_release_allows_retry(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(d, "_redis", lambda: fake)
    assert d.claim_location_event_id(7, "evt-abc") is True
    assert d.claim_location_event_id(7, "evt-abc") is False
    d.release_location_event_id(7, "evt-abc")
    assert d.claim_location_event_id(7, "evt-abc") is True


def test_should_skip_after_release_is_false(monkeypatch) -> None:
    from datetime import UTC, datetime

    fake = _FakeRedis()
    monkeypatch.setattr(d, "_redis", lambda: fake)
    monkeypatch.setattr(d, "should_skip_proximity_duplicate", lambda *_a, **_k: False)
    now = datetime.now(UTC)
    skip1, _reason1 = d.should_skip_location_ingest(
        1, 46.0, 6.0, now, "availability_presence", "evt-1"
    )
    assert skip1 is False
    skip2, reason2 = d.should_skip_location_ingest(
        1, 46.0, 6.0, now, "availability_presence", "evt-1"
    )
    assert skip2 is True
    assert reason2 == "duplicate_event_id"
    d.release_location_event_id(1, "evt-1")
    skip3, reason3 = d.should_skip_location_ingest(
        1, 46.0, 6.0, now, "availability_presence", "evt-1"
    )
    assert skip3 is False
    assert reason3 is None
