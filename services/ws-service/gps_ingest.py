"""Buffer GPS ws-service → POST backend internal ingest (batch, rate-limit, F-01)."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections import deque
from datetime import UTC, datetime, timedelta
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
MAX_BUFFER_POINTS = int(os.getenv("WS_GPS_MAX_BUFFER_POINTS", "2000"))
CIRCUIT_OPEN_SEC = float(os.getenv("WS_GPS_CIRCUIT_OPEN_SEC", "60"))
MAX_BACKOFF_SEC = float(os.getenv("WS_GPS_MAX_BACKOFF_SEC", "30"))
# Marge d'horloge vs skew backend passé 24 h
PURGE_MAX_AGE = timedelta(hours=23, minutes=50)

_lock = threading.RLock()
# Buffer global FIFO : (driver_id, point, enqueued_at_mono)
_buffer: deque[tuple[int, dict[str, Any], float]] = deque()
_last_flush_mono = 0.0
_request_times: dict[int, list[float]] = {}
_queue_depth = 0
_dropped_points = 0
_dropped_oldest = 0
_purged_skew = 0
_ingest_requests = 0
_retry_total = 0

# Circuit breaker : closed | open | half_open
_circuit_state = "closed"
_circuit_opened_at = 0.0
_half_open_in_flight = False
_backoff_until = 0.0


def _service_token() -> str:
    return (os.getenv("INTERNAL_SERVICE_TOKEN") or "").strip()


def stats() -> dict[str, int | str]:
    with _lock:
        return {
            "queue_depth": _queue_depth,
            "dropped_points": _dropped_points,
            "dropped_oldest": _dropped_oldest,
            "purged_skew": _purged_skew,
            "ingest_requests": _ingest_requests,
            "retry_total": _retry_total,
            "circuit_state": _circuit_state,
        }


def _rate_ok(driver_id: int) -> bool:
    now = time.time()
    window = [t for t in _request_times.get(driver_id, []) if now - t < 60]
    _request_times[driver_id] = window
    return len(window) < RATE_LIMIT_PER_MIN


def _recompute_depth() -> None:
    global _queue_depth
    _queue_depth = len(_buffer)


def _evict_oldest(count: int = 1) -> None:
    global _dropped_points, _dropped_oldest
    for _ in range(count):
        if not _buffer:
            break
        _buffer.popleft()
        _dropped_points += 1
        _dropped_oldest += 1
    if count > 0:
        logger.warning(
            "gps buffer overflow drop_oldest count=%s depth=%s",
            count,
            len(_buffer),
        )


def _parse_point_time(point: dict[str, Any]) -> datetime | None:
    raw = point.get("recorded_at") or point.get("timestamp")
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        try:
            ts = float(raw)
            if ts > 1e12:
                ts /= 1000.0
            return datetime.fromtimestamp(ts, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(raw, str):
        return None
    text = raw.strip().replace("Z", "+00:00") if raw.endswith("Z") else raw.strip()
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _purge_skew_locked() -> int:
    """Supprime les points plus anciens que 23 h 50 (avant constitution batch)."""
    global _purged_skew
    now = datetime.now(UTC)
    cutoff = now - PURGE_MAX_AGE
    kept: deque[tuple[int, dict[str, Any], float]] = deque()
    removed = 0
    while _buffer:
        driver_id, point, mono = _buffer.popleft()
        pt = _parse_point_time(point)
        if pt is not None and pt < cutoff:
            removed += 1
            continue
        kept.append((driver_id, point, mono))
    _buffer.extend(kept)
    if removed:
        _purged_skew += removed
        logger.info("gps purge skew removed=%s remaining=%s", removed, len(_buffer))
    return removed


def enqueue_point(driver_id: int, point: dict[str, Any]) -> None:
    """Ajoute un point. ``driver_id`` doit venir de la session socket (jamais du payload)."""
    global _dropped_points
    if not isinstance(point, dict):
        return
    # Ne jamais faire confiance à un driver_id fourni par le mobile.
    safe_point = {k: v for k, v in point.items() if k != "driver_id"}
    with _lock:
        _buffer.append((driver_id, safe_point, time.monotonic()))
        overflow = len(_buffer) - MAX_BUFFER_POINTS
        if overflow > 0:
            _evict_oldest(overflow)
        _recompute_depth()


def _circuit_allows_request() -> bool:
    global _circuit_state, _half_open_in_flight
    now = time.monotonic()
    if _circuit_state == "closed":
        return True
    if _circuit_state == "open":
        if now - _circuit_opened_at < CIRCUIT_OPEN_SEC:
            return False
        _circuit_state = "half_open"
        _half_open_in_flight = False
    if _circuit_state == "half_open":
        if _half_open_in_flight:
            return False
        _half_open_in_flight = True
        return True
    return True


def _open_circuit() -> None:
    global _circuit_state, _circuit_opened_at, _half_open_in_flight
    _circuit_state = "open"
    _circuit_opened_at = time.monotonic()
    _half_open_in_flight = False
    logger.critical(
        "gps ingest circuit OPEN (401/403) — retry probe in %ss", CIRCUIT_OPEN_SEC
    )


def _close_circuit() -> None:
    global _circuit_state, _half_open_in_flight
    _circuit_state = "closed"
    _half_open_in_flight = False
    logger.info("gps ingest circuit CLOSED")


def _requeue_front(items: list[tuple[int, dict[str, Any], float]]) -> None:
    for item in reversed(items):
        _buffer.appendleft(item)
    overflow = len(_buffer) - MAX_BUFFER_POINTS
    if overflow > 0:
        _evict_oldest(overflow)
    _recompute_depth()


def _take_batch() -> list[tuple[int, dict[str, Any], float]]:
    """Extrait jusqu'à MAX_POINTS_PER_BATCH points du même chauffeur (tête FIFO)."""
    _purge_skew_locked()
    if not _buffer:
        return []
    first_driver = _buffer[0][0]
    batch: list[tuple[int, dict[str, Any], float]] = []
    while _buffer and len(batch) < MAX_POINTS_PER_BATCH and _buffer[0][0] == first_driver:
        batch.append(_buffer.popleft())
    _recompute_depth()
    return batch


async def flush_once() -> None:
    """Flush un lot (pour tests / boucle)."""
    global _ingest_requests, _retry_total, _last_flush_mono, _backoff_until
    global _half_open_in_flight

    with _lock:
        if time.monotonic() < _backoff_until:
            return
        if not _circuit_allows_request():
            return
        batch = _take_batch()
        if not batch:
            if _circuit_state == "half_open":
                _half_open_in_flight = False
            return
        driver_id = batch[0][0]
        if not _rate_ok(driver_id):
            _requeue_front(batch)
            if _circuit_state == "half_open":
                _half_open_in_flight = False
            return

    token = _service_token()
    points = [p for _, p, _ in batch]
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "X-Internal-Service": os.getenv("INTERNAL_SERVICE_AUDIENCE", "ws-service"),
    }
    if not token:
        logger.error(
            "gps ingest aborted: INTERNAL_SERVICE_TOKEN missing (F-01 fail-closed)"
        )
        with _lock:
            _retry_total += 1
            _requeue_front(batch)
            if _circuit_state == "half_open":
                _half_open_in_flight = False
            _backoff_until = time.monotonic() + min(5.0, MAX_BACKOFF_SEC)
        return
    headers["X-Internal-Token"] = token

    body = {"driver_id": driver_id, "points": points}
    try:
        async with httpx.AsyncClient(timeout=INGEST_TIMEOUT_SEC) as client:
            resp = await client.post(INGEST_URL, json=body, headers=headers)
        with _lock:
            _request_times.setdefault(driver_id, []).append(time.time())
            _ingest_requests += 1
            _last_flush_mono = time.monotonic()
            status = resp.status_code
            if 200 <= status < 300:
                _close_circuit()
                return
            if status in (401, 403):
                _retry_total += 1
                _requeue_front(batch)
                _open_circuit()
                logger.warning("gps ingest http %s driver_id=%s", status, driver_id)
                return
            if status == 400:
                _retry_total += 1
                logger.warning(
                    "gps ingest drop deterministic 400 driver_id=%s body=%s",
                    driver_id,
                    (resp.text or "")[:200],
                )
                if _circuit_state == "half_open":
                    _half_open_in_flight = False
                return
            # 429 / 5xx
            _retry_total += 1
            _requeue_front(batch)
            delay = min(MAX_BACKOFF_SEC, 1.0 + (_retry_total % 10))
            _backoff_until = time.monotonic() + delay
            if _circuit_state == "half_open":
                # rester half-open ; prochaine tentative après backoff
                _half_open_in_flight = False
            logger.warning("gps ingest http %s driver_id=%s requeue", status, driver_id)
    except Exception:
        with _lock:
            _retry_total += 1
            _requeue_front(batch)
            _backoff_until = time.monotonic() + min(MAX_BACKOFF_SEC, 2.0)
            if _circuit_state == "half_open":
                _half_open_in_flight = False
        logger.exception("gps ingest failed driver_id=%s", driver_id)


async def flush_driver(driver_id: int) -> None:
    """Compat : déclenche un flush (le batch est FIFO multi-chauffeurs)."""
    _ = driver_id
    await flush_once()


async def flush_loop() -> None:
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_SEC)
        # Plusieurs lots si buffer plein
        for _ in range(20):
            with _lock:
                empty = len(_buffer) == 0
            if empty:
                break
            await flush_once()
