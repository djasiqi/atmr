"""Tests P0-4 processed.v3 Lua — gen/seq/event_id/hash."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from processed_apply import (
    APPLIED_NEW,
    DUPLICATE_CURRENT,
    EVENT_ID_PAYLOAD_CONFLICT,
    SEQUENCE_EVENT_CONFLICT,
    STALE_OLDER,
    apply_processed_canonical,
)


class _FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, dict[str, str]] = {}
        self.eval = AsyncMock(side_effect=self._eval)

    async def _eval(self, script: str, numkeys: int, *args):
        key = args[0]
        in_eid, in_hash = args[1], args[2]
        in_gen, in_seq = int(args[3]), int(args[4])
        lat, lon, recorded_at, company_id = args[5], args[6], args[7], args[8]
        cur = self.store.get(key)
        if not cur:
            self.store[key] = {
                "location_event_id": in_eid,
                "event_payload_hash": in_hash,
                "session_generation": str(in_gen),
                "sequence_id": str(in_seq),
                "lat": lat,
                "lon": lon,
                "recorded_at": recorded_at,
                "company_id": company_id,
            }
            return APPLIED_NEW
        cur_gen = int(cur.get("session_generation") or -1)
        cur_seq = int(cur.get("sequence_id") or -1)
        cur_eid = cur.get("location_event_id") or ""
        cur_hash = cur.get("event_payload_hash") or ""
        if in_gen < cur_gen or (in_gen == cur_gen and in_seq < cur_seq):
            return STALE_OLDER
        if in_gen == cur_gen and in_seq == cur_seq:
            if in_eid != cur_eid:
                return SEQUENCE_EVENT_CONFLICT
            if in_hash != cur_hash:
                return EVENT_ID_PAYLOAD_CONFLICT
            return DUPLICATE_CURRENT
        if in_eid == cur_eid and in_hash != cur_hash:
            return EVENT_ID_PAYLOAD_CONFLICT
        self.store[key] = {
            "location_event_id": in_eid,
            "event_payload_hash": in_hash,
            "session_generation": str(in_gen),
            "sequence_id": str(in_seq),
            "lat": lat,
            "lon": lon,
            "recorded_at": recorded_at,
            "company_id": company_id,
        }
        return APPLIED_NEW


def _run(**kwargs):
    redis = kwargs.pop("redis")
    emit = kwargs.pop("emit", AsyncMock())
    return asyncio.run(
        apply_processed_canonical(
            redis,
            emit_fn=emit,
            company_room_fn=lambda c: f"company_{c}",
            driver_room_fn=lambda d: f"driver_{d}",
            **kwargs,
        )
    ), emit


def test_processed_applied_new_and_fanout():
    redis = _FakeRedis()
    code, emit = _run(
        redis=redis,
        driver_id=1,
        location_event_id="e1",
        event_payload_hash="h1",
        session_generation=1,
        sequence_id=1,
        latitude=46.0,
        longitude=6.0,
        recorded_at="t",
        company_id=9,
        payload={"foo": 1},
    )
    assert code == APPLIED_NEW
    emit.assert_called_once()


def test_processed_duplicate_refanout():
    redis = _FakeRedis()
    kwargs = dict(
        redis=redis,
        driver_id=1,
        location_event_id="e1",
        event_payload_hash="h1",
        session_generation=1,
        sequence_id=10,
        latitude=46.0,
        longitude=6.0,
        recorded_at="t",
        company_id=9,
        payload={},
    )
    _run(**kwargs)
    code, emit = _run(**kwargs)
    assert code == DUPLICATE_CURRENT
    emit.assert_called_once()


def test_processed_stale_older_no_fanout():
    redis = _FakeRedis()
    _run(
        redis=redis,
        driver_id=1,
        location_event_id="e2",
        event_payload_hash="h2",
        session_generation=2,
        sequence_id=5,
        latitude=1.0,
        longitude=2.0,
        recorded_at="t",
        company_id=1,
        payload={},
    )
    code, emit = _run(
        redis=redis,
        driver_id=1,
        location_event_id="e1",
        event_payload_hash="h1",
        session_generation=1,
        sequence_id=1,
        latitude=1.0,
        longitude=2.0,
        recorded_at="t",
        company_id=1,
        payload={},
    )
    assert code == STALE_OLDER
    emit.assert_not_called()


def test_sequence_event_conflict():
    redis = _FakeRedis()
    _run(
        redis=redis,
        driver_id=1,
        location_event_id="A",
        event_payload_hash="h",
        session_generation=7,
        sequence_id=100,
        latitude=1.0,
        longitude=2.0,
        recorded_at="t",
        company_id=1,
        payload={},
    )
    code, emit = _run(
        redis=redis,
        driver_id=1,
        location_event_id="B",
        event_payload_hash="h",
        session_generation=7,
        sequence_id=100,
        latitude=1.0,
        longitude=2.0,
        recorded_at="t",
        company_id=1,
        payload={},
    )
    assert code == SEQUENCE_EVENT_CONFLICT
    emit.assert_not_called()


def test_event_id_payload_conflict():
    redis = _FakeRedis()
    _run(
        redis=redis,
        driver_id=1,
        location_event_id="A",
        event_payload_hash="h1",
        session_generation=7,
        sequence_id=100,
        latitude=1.0,
        longitude=2.0,
        recorded_at="t",
        company_id=1,
        payload={},
    )
    code, emit = _run(
        redis=redis,
        driver_id=1,
        location_event_id="A",
        event_payload_hash="h2",
        session_generation=7,
        sequence_id=100,
        latitude=1.0,
        longitude=2.0,
        recorded_at="t",
        company_id=1,
        payload={},
    )
    assert code == EVENT_ID_PAYLOAD_CONFLICT
    emit.assert_not_called()
