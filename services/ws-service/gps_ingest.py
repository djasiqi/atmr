"""Buffer GPS ws-service → POST backend internal ingest (batch, rate-limit)."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict
from typing import Any

import httpx

logger = logging.getLogger("ws-service.gps_ingest")

FLUSH_INTERVAL_SEC = float(os.getenv("WS_GPS_FLUSH_INTERVAL_SEC", "3"))
MAX_POINTS_PER_BATCH = int(os.getenv("WS_GPS_MAX_POINTS_PER_BATCH", "10"))
INGEST_TIMEOUT_SEC = float(os.getenv("WS_GPS_INGEST_TIMEOUT_SEC", "0.2"))
RATE_LIMIT_PER_MIN = int(os.getenv("WS_GPS_RATE_LIMIT_PER_DRIVER_PER_MIN", "30"))
INGEST_URL = os.getenv(
    "BACKEND_INTERNAL_TRACKING_INGEST_URL",
    "http://backend:5000/api/internal/tracking/ingest",
)
SERVICE_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "")

_buffers: dict[int, list[dict[str, Any]]] = defaultdict(list)
_last_flush: dict[int, float] = defaultdict(float)
_request_times: dict[int, list[float]] = defaultdict(list)
_queue_depth = 0
_dropped_points = 0
_ingest_requests = 0
_retry_total = 0


def stats() -> dict[str, int]:
    return {
        "queue_depth": _queue_depth,
        "dropped_points": _dropped_points,
        "ingest_requests": _ingest_requests,
        "retry_total": _retry_total,
    }


def _rate_ok(driver_id: int) -> bool:
    now = time.time()
    window = [t for t in _request_times[driver_id] if now - t < 60]
    _request_times[driver_id] = window
    return len(window) < RATE_LIMIT_PER_MIN


def enqueue_point(driver_id: int, point: dict[str, Any]) -> None:
    global _queue_depth, _dropped_points
    _buffers[driver_id].append(point)
    _queue_depth = sum(len(v) for v in _buffers.values())
    if len(_buffers[driver_id]) > MAX_POINTS_PER_BATCH * 3:
        dropped = _buffers[driver_id].pop(0)
        _dropped_points += 1
        logger.warning("gps buffer drop driver_id=%s point=%s", driver_id, dropped)


async def flush_driver(driver_id: int) -> None:
    global _ingest_requests, _retry_total, _queue_depth
    points = _buffers.pop(driver_id, [])
    if not points:
        return
    if not _rate_ok(driver_id):
        _buffers[driver_id] = points + _buffers[driver_id]
        return

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if SERVICE_TOKEN:
        headers["X-Internal-Token"] = SERVICE_TOKEN

    body = {"driver_id": driver_id, "points": points}
    try:
        async with httpx.AsyncClient(timeout=INGEST_TIMEOUT_SEC) as client:
            resp = await client.post(INGEST_URL, json=body, headers=headers)
        _request_times[driver_id].append(time.time())
        _ingest_requests += 1
        if resp.status_code >= 400:
            _retry_total += 1
            logger.warning(
                "gps ingest http %s driver_id=%s", resp.status_code, driver_id
            )
    except Exception:
        _retry_total += 1
        logger.exception("gps ingest failed driver_id=%s", driver_id)
    finally:
        _queue_depth = sum(len(v) for v in _buffers.values())
        _last_flush[driver_id] = time.time()


async def flush_loop() -> None:
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_SEC)
        now = time.time()
        for driver_id in list(_buffers.keys()):
            if (
                len(_buffers[driver_id]) >= MAX_POINTS_PER_BATCH
                or now - _last_flush[driver_id] >= FLUSH_INTERVAL_SEC
            ):
                await flush_driver(driver_id)
