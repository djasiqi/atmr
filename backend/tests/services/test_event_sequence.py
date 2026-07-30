"""Tests séquence Redis — curseur None si dégradé."""

from __future__ import annotations

import ext
from services.realtime.event_sequence import (
    current_snapshot_cursor,
    get_snapshot_cursor_status,
    next_event_seq,
)


class _FakeRedis:
    def __init__(self) -> None:
        self._store: dict[str, int] = {}

    def incr(self, key: str) -> int:
        self._store[key] = self._store.get(key, 0) + 1
        return self._store[key]

    def get(self, key: str) -> bytes | None:
        val = self._store.get(key)
        return str(val).encode("utf-8") if val is not None else None


def test_next_event_seq_increments_per_company(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ext, "redis_client", fake)

    assert next_event_seq(42) == 1
    assert next_event_seq(42) == 2
    assert next_event_seq(43) == 1


def test_current_snapshot_cursor_reflects_last_event_seq(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ext, "redis_client", fake)

    assert current_snapshot_cursor(7) == 0
    next_event_seq(7)
    next_event_seq(7)
    assert current_snapshot_cursor(7) == 2


def test_events_after_bootstrap_are_strictly_greater_than_cursor(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ext, "redis_client", fake)

    next_event_seq(99)
    snapshot_cursor = current_snapshot_cursor(99)
    seq_after = next_event_seq(99)
    assert seq_after > snapshot_cursor


def test_redis_unavailable_returns_none_degraded(monkeypatch) -> None:
    monkeypatch.setattr(ext, "redis_client", None)
    assert next_event_seq(1) == 0
    assert current_snapshot_cursor(1) is None
    cursor, status = get_snapshot_cursor_status(1)
    assert cursor is None
    assert status == "degraded"


def test_next_event_seq_fail_open_without_company_id(monkeypatch) -> None:
    fake = _FakeRedis()
    monkeypatch.setattr(ext, "redis_client", fake)
    assert next_event_seq(None) == 0
    assert next_event_seq(0) == 0
