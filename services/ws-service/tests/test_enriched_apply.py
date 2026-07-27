"""Phase 3 : enriched.v3 n'applique Redis que si location_event_id courant."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from enriched_apply import apply_enriched_canonical


def test_apply_enriched_stale_skipped():
    redis = AsyncMock()
    redis.hgetall = AsyncMock(
        return_value={"location_event_id": "current-event", "lat": "46.0", "lon": "6.0"}
    )
    redis.hset = AsyncMock()
    emit = AsyncMock()

    applied = asyncio.run(
        apply_enriched_canonical(
            redis,
            driver_id=1,
            payload={
                "canonical_latitude": 46.1,
                "canonical_longitude": 6.1,
                "company_id": 9,
            },
            location_event_id="stale-event",
            emit_fn=emit,
            company_room_fn=lambda cid: f"company_{cid}",
            driver_room_fn=lambda did: f"driver_{did}",
        )
    )
    assert applied is False
    redis.hset.assert_not_called()


def test_apply_enriched_current_updates():
    redis = AsyncMock()
    redis.hgetall = AsyncMock(
        return_value={"location_event_id": "e-ok", "lat": "46.0", "lon": "6.0"}
    )
    redis.hset = AsyncMock()
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
            },
            location_event_id="e-ok",
            emit_fn=emit,
            company_room_fn=lambda cid: f"company_{cid}",
            driver_room_fn=lambda did: f"driver_{did}",
        )
    )
    assert applied is True
    redis.hset.assert_called_once()
    emit.assert_called_once()


def test_two_pods_fanout_uses_emit_not_local_only():
    """Gate 2 pods : l'application enriched passe par emit_fn (AsyncRedisManager)."""
    redis = AsyncMock()
    redis.hgetall = AsyncMock(
        return_value={"location_event_id": "e1", "lat": "1", "lon": "2"}
    )
    redis.hset = AsyncMock()
    emit = AsyncMock()
    asyncio.run(
        apply_enriched_canonical(
            redis,
            driver_id=3,
            payload={"canonical_latitude": 1.1, "canonical_longitude": 2.2, "company_id": 1},
            location_event_id="e1",
            emit_fn=emit,
            company_room_fn=lambda cid: f"company_{cid}",
            driver_room_fn=lambda did: f"driver_{did}",
        )
    )
    assert emit.await_count == 1
    assert emit.await_args.args[0] == "driver_location_update"
