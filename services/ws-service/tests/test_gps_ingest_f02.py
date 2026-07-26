"""Tests F-02 — ACK durable, 409→DLQ, batch_id, refus saturation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import gps_ingest as gi


def setup_function() -> None:
    with gi._lock:
        gi._buffer.clear()
        gi._request_times.clear()
        gi._queue_depth = 0
        gi._dropped_points = 0
        gi._dropped_oldest = 0
        gi._purged_skew = 0
        gi._ingest_requests = 0
        gi._retry_total = 0
        gi._rejected_total = 0
        gi._dlq_total = 0
        gi._acked_total = 0
        gi._spooled_total = 0
        gi._circuit_state = "closed"
        gi._circuit_opened_at = 0.0
        gi._half_open_in_flight = False
        gi._backoff_until = 0.0
    gi.SPOOL_BACKEND = "memory"
    gi.FLUSH_ENABLED = True


def _point(**kw):
    now = datetime.now(UTC).isoformat()
    base = {
        "latitude": 46.2,
        "longitude": 6.1,
        "recorded_at": now,
        "location_event_id": "e1",
        "location_mode": "mission_live",
        "company_id": 10,
    }
    base.update(kw)
    return base


def test_409_goes_to_dlq_no_requeue(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "x" * 32)
    gi.enqueue_point(7, _point())
    mock_resp = MagicMock(status_code=409, text='{"error_code":"event_id_payload_conflict"}')
    mock_resp.json.return_value = {
        "error_code": "event_id_payload_conflict",
        "conflicting_event_ids": ["e1"],
    }

    async def _run() -> None:
        with patch("httpx.AsyncClient") as client_cls:
            instance = AsyncMock()
            instance.__aenter__.return_value = instance
            instance.__aexit__.return_value = None
            instance.post = AsyncMock(return_value=mock_resp)
            client_cls.return_value = instance
            await gi.flush_once()

    asyncio.run(_run())
    with gi._lock:
        assert len(gi._buffer) == 0
    assert gi.stats()["dlq_total"] >= 1


def test_200_incomplete_json_requeues(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "x" * 32)
    gi.enqueue_point(7, _point())
    mock_resp = MagicMock(status_code=200, text="{}")
    mock_resp.json.return_value = {"ok": True}  # missing durability / batch_id

    async def _run() -> None:
        with patch("httpx.AsyncClient") as client_cls:
            instance = AsyncMock()
            instance.__aenter__.return_value = instance
            instance.__aexit__.return_value = None
            instance.post = AsyncMock(return_value=mock_resp)
            client_cls.return_value = instance
            await gi.flush_once()

    asyncio.run(_run())
    with gi._lock:
        assert len(gi._buffer) == 1
    assert gi.stats()["retry_total"] >= 1


def test_200_durable_acks(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "x" * 32)
    pt = _point()
    gi.enqueue_point(7, pt)

    def _post_side_effect(*args, **kwargs):
        body = kwargs.get("json") or {}
        batch_id = body.get("batch_id") or ("c" * 64)
        resp = MagicMock(status_code=200)
        resp.json.return_value = {
            "ok": True,
            "batch_id": batch_id,
            "durability": "postgres_committed",
            "received": 1,
            "persisted": 1,
            "duplicates": 0,
        }
        resp.text = "{}"
        return resp

    async def _run() -> None:
        with patch("httpx.AsyncClient") as client_cls:
            instance = AsyncMock()
            instance.__aenter__.return_value = instance
            instance.__aexit__.return_value = None
            instance.post = AsyncMock(side_effect=_post_side_effect)
            client_cls.return_value = instance
            await gi.flush_once()

    asyncio.run(_run())
    with gi._lock:
        assert len(gi._buffer) == 0
    assert gi.stats()["acked_total"] >= 1
