"""Tests P0-2 / P0-3 gates ws-service Kafka primary."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture()
def kafka_primary_env(monkeypatch):
    monkeypatch.setenv("TRACKING_INGEST_MODE", "kafka_primary")
    monkeypatch.setenv("WS_KAFKA_CONSUMER_ENABLED", "true")


def test_emit_tracking_bypasses_deduper(kafka_primary_env):
    import importlib
    import main as ws_main

    importlib.reload(ws_main)
    assert ws_main._IS_KAFKA_PRIMARY is True

    emitted: list[tuple] = []

    async def fake_emit(event_type, payload, to=None):
        emitted.append((event_type, payload, to))

    ws_main.sio.emit = fake_emit  # type: ignore[method-assign]
    # Deduper would block if called — ensure tracking path doesn't use it
    with patch.object(ws_main.deduper, "should_emit", return_value=False) as should:
        import asyncio

        asyncio.run(
            ws_main._emit_tracking_to_room(
                "driver_location_update",
                {"location_event_id": "e1", "event_id": "e1"},
                "company_1",
                user_id="42",
            )
        )
        should.assert_not_called()
    assert len(emitted) == 1


def test_commit_message_exact_partition():
    import asyncio

    import main as ws_main

    committed: dict = {}

    class Msg:
        topic = "driver.location.processed.v3"
        partition = 2
        offset = 41

    class Consumer:
        async def commit(self, offsets):
            committed.update(offsets)

    with patch.dict(os.environ, {"TRACKING_INGEST_MODE": "kafka_primary"}):
        asyncio.run(ws_main._commit_message(Consumer(), Msg()))

    assert len(committed) == 1
    tp = next(iter(committed))
    assert tp.partition == 2
    assert committed[tp].offset == 42


def test_fail_stop_engages_kill_switch():
    import main as ws_main

    with patch.object(ws_main, "engage_kill_switch") as engage:
        with pytest.raises(ws_main.FatalRealtimeConsumerError):
            ws_main._fail_stop_consumer("redis_ko")
        engage.assert_called_once()
