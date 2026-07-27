"""Phase 3 : enriched.v3 Lua — versions séparées + conflits."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from enriched_apply import (
    APPLIED_NEW_VERSION,
    DUPLICATE_CURRENT_VERSION,
    STALE_EVENT,
    STALE_VERSION,
    apply_enriched_canonical,
)


class _FakeRedis:
    """Redis minimal pour eval Lua (simulation côté tests unitaires)."""

    def __init__(self, store: dict[str, dict[str, str]] | None = None) -> None:
        self.store = store or {}
        self.hset = AsyncMock(side_effect=self._hset)
        self.eval = AsyncMock(side_effect=self._eval)

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.store.get(key, {}))

    async def _hset(self, key: str, mapping: dict[str, str] | None = None, **kwargs):
        self.store.setdefault(key, {}).update(mapping or kwargs)

    async def _eval(self, script: str, numkeys: int, *args):
        keys = list(args[:numkeys])
        argv = list(args[numkeys:])
        key = keys[0]
        legacy = keys[1] if len(keys) > 1 else key
        in_eid, in_ver, lat, lon, source = argv[0], int(argv[1]), argv[2], argv[3], argv[4]
        cur = self.store.get(key) or {}
        if not cur:
            cur = self.store.get(legacy) or {}
            if not cur:
                return STALE_EVENT
            key = legacy
        if str(cur.get("location_event_id") or "") != in_eid:
            return STALE_EVENT
        cur_ver = int(cur.get("enrichment_version") or 0)
        if in_ver < cur_ver:
            return STALE_VERSION
        if in_ver == cur_ver:
            return DUPLICATE_CURRENT_VERSION
        self.store[key] = {
            **cur,
            "lat": lat,
            "lon": lon,
            "canonical_latitude": lat,
            "canonical_longitude": lon,
            "canonical_source": source,
            "enrichment_version": str(in_ver),
            "location_event_id": in_eid,
        }
        return APPLIED_NEW_VERSION


def test_apply_enriched_stale_skipped():
    redis = _FakeRedis(
        {"driver:1:loc:canonical": {"location_event_id": "current-event", "lat": "46.0", "lon": "6.0"}}
    )
    emit = AsyncMock()
    applied = asyncio.run(
        apply_enriched_canonical(
            redis,
            driver_id=1,
            payload={
                "canonical_latitude": 46.1,
                "canonical_longitude": 6.1,
                "company_id": 9,
                "enrichment_version": 1,
            },
            location_event_id="stale-event",
            emit_fn=emit,
            company_room_fn=lambda cid: f"company_{cid}",
            driver_room_fn=lambda did: f"driver_{did}",
        )
    )
    assert applied == STALE_EVENT
    emit.assert_not_called()


def test_apply_enriched_current_updates():
    redis = _FakeRedis(
        {
            "driver:1:loc:canonical": {
                "location_event_id": "e-ok",
                "lat": "46.0",
                "lon": "6.0",
                "enrichment_version": "0",
            }
        }
    )
    emit = AsyncMock()
    applied = asyncio.run(
        apply_enriched_canonical(
            redis,
            driver_id=1,
            payload={
                "canonical_latitude": 46.1,
                "canonical_longitude": 6.1,
                "company_id": 9,
                "canonical_source": "osrm",
                "enrichment_version": 1,
            },
            location_event_id="e-ok",
            emit_fn=emit,
            company_room_fn=lambda cid: f"company_{cid}",
            driver_room_fn=lambda did: f"driver_{did}",
        )
    )
    assert applied == APPLIED_NEW_VERSION
    emit.assert_called_once()


def test_enriched_v2_then_v1_stale_version():
    redis = _FakeRedis(
        {
            "driver:1:loc:canonical": {
                "location_event_id": "e1",
                "enrichment_version": "2",
                "lat": "1",
                "lon": "2",
            }
        }
    )
    emit = AsyncMock()
    code = asyncio.run(
        apply_enriched_canonical(
            redis,
            driver_id=1,
            payload={
                "canonical_latitude": 1.1,
                "canonical_longitude": 2.2,
                "enrichment_version": 1,
            },
            location_event_id="e1",
            emit_fn=emit,
        )
    )
    assert code == STALE_VERSION
    emit.assert_not_called()


def test_enriched_duplicate_refanout():
    redis = _FakeRedis(
        {
            "driver:1:loc:canonical": {
                "location_event_id": "e1",
                "enrichment_version": "2",
                "lat": "1",
                "lon": "2",
            }
        }
    )
    emit = AsyncMock()
    code = asyncio.run(
        apply_enriched_canonical(
            redis,
            driver_id=1,
            payload={
                "canonical_latitude": 1.1,
                "canonical_longitude": 2.2,
                "company_id": 1,
                "enrichment_version": 2,
            },
            location_event_id="e1",
            emit_fn=emit,
            company_room_fn=lambda cid: f"company_{cid}",
            driver_room_fn=lambda did: f"driver_{did}",
        )
    )
    assert code == DUPLICATE_CURRENT_VERSION
    assert emit.await_count == 1


def test_two_pods_fanout_uses_emit_not_local_only():
    """Gate 2 pods : l'application enriched passe par emit_fn (AsyncRedisManager)."""
    redis = _FakeRedis(
        {
            "driver:3:loc:canonical": {
                "location_event_id": "e1",
                "lat": "1",
                "lon": "2",
                "enrichment_version": "0",
            }
        }
    )
    emit = AsyncMock()
    asyncio.run(
        apply_enriched_canonical(
            redis,
            driver_id=3,
            payload={
                "canonical_latitude": 1.1,
                "canonical_longitude": 2.2,
                "company_id": 1,
                "enrichment_version": 1,
            },
            location_event_id="e1",
            emit_fn=emit,
            company_room_fn=lambda cid: f"company_{cid}",
            driver_room_fn=lambda did: f"driver_{did}",
        )
    )
    assert emit.await_count == 1
    assert emit.await_args.args[0] == "driver_location_update"
