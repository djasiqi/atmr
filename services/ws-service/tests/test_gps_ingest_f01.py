"""Tests F-01 — gps_ingest ws-service (circuit, buffer, binding)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
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
        gi._circuit_state = "closed"
        gi._circuit_opened_at = 0.0
        gi._half_open_in_flight = False
        gi._backoff_until = 0.0


def test_binding_strips_payload_driver_id() -> None:
    gi.enqueue_point(42, {"driver_id": 999, "latitude": 1.0, "longitude": 2.0})
    with gi._lock:
        assert len(gi._buffer) == 1
        driver_id, point, _ = gi._buffer[0]
    assert driver_id == 42
    assert "driver_id" not in point


def test_buffer_global_max_drops_oldest() -> None:
    old_max = gi.MAX_BUFFER_POINTS
    try:
        gi.MAX_BUFFER_POINTS = 3
        for i in range(5):
            gi.enqueue_point(1, {"latitude": float(i), "recorded_at": "2026-01-01T00:00:00Z"})
        with gi._lock:
            assert len(gi._buffer) == 3
            assert gi._buffer[0][1]["latitude"] == 2.0
        assert gi.stats()["dropped_oldest"] == 2
    finally:
        gi.MAX_BUFFER_POINTS = old_max


def test_purge_skew_before_batch() -> None:
    old = (datetime.now(UTC) - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    gi.enqueue_point(1, {"recorded_at": old, "latitude": 1.0})
    gi.enqueue_point(1, {"recorded_at": recent, "latitude": 2.0})
    with gi._lock:
        removed = gi._purge_skew_locked()
        assert removed == 1
        assert len(gi._buffer) == 1
        assert gi._buffer[0][1]["latitude"] == 2.0


def test_abort_without_token_requeues(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "")
    gi.enqueue_point(7, {"latitude": 1.0, "longitude": 2.0, "recorded_at": datetime.now(UTC).isoformat()})

    async def _run() -> None:
        await gi.flush_once()

    asyncio.run(_run())
    with gi._lock:
        assert len(gi._buffer) == 1
    assert gi.stats()["retry_total"] >= 1


def test_retry_on_503(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "x" * 32)
    gi.enqueue_point(
        7,
        {
            "latitude": 1.0,
            "longitude": 2.0,
            "recorded_at": datetime.now(UTC).isoformat(),
        },
    )
    mock_resp = MagicMock(status_code=503, text="ingest_disabled")

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


def test_circuit_opens_on_401(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "x" * 32)
    gi.enqueue_point(
        7,
        {
            "latitude": 1.0,
            "longitude": 2.0,
            "recorded_at": datetime.now(UTC).isoformat(),
        },
    )
    mock_resp = MagicMock(status_code=401, text="unauthorized")

    async def _run() -> None:
        with patch("httpx.AsyncClient") as client_cls:
            instance = AsyncMock()
            instance.__aenter__.return_value = instance
            instance.__aexit__.return_value = None
            instance.post = AsyncMock(return_value=mock_resp)
            client_cls.return_value = instance
            await gi.flush_once()

    asyncio.run(_run())
    assert gi.stats()["circuit_state"] == "open"
    with gi._lock:
        assert len(gi._buffer) == 1


def test_drop_on_400(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "x" * 32)
    gi.enqueue_point(
        7,
        {
            "latitude": 1.0,
            "longitude": 2.0,
            "recorded_at": datetime.now(UTC).isoformat(),
        },
    )
    mock_resp = MagicMock(status_code=400, text="invalid")

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
