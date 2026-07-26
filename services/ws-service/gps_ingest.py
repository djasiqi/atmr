"""Buffer / spool GPS ws-service → POST backend internal ingest (F-02 ACK durable).

Backends : ``WS_GPS_SPOOL_BACKEND=memory`` (défaut tests) ou ``redis_stream``.
ACK spool uniquement si HTTP 200 + contrat durabilité PostgreSQL complet.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

import event_payload_hash as eph
import gps_spool as spool

logger = logging.getLogger("ws-service.gps_ingest")

SPOOL_BACKEND = os.getenv("WS_GPS_SPOOL_BACKEND", "memory").strip().lower()
FLUSH_ENABLED = os.getenv("WS_GPS_FLUSH_ENABLED", "true").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

FLUSH_INTERVAL_SEC = float(os.getenv("WS_GPS_FLUSH_INTERVAL_SEC", "3"))
MAX_POINTS_PER_BATCH = int(os.getenv("WS_GPS_MAX_POINTS_PER_BATCH", "10"))
RATE_LIMIT_PER_MIN = int(os.getenv("WS_GPS_RATE_LIMIT_PER_DRIVER_PER_MIN", "30"))
INGEST_URL = os.getenv(
    "BACKEND_INTERNAL_TRACKING_INGEST_URL",
    "http://backend:5000/api/internal/tracking/ingest",
)
MAX_BUFFER_POINTS = int(os.getenv("WS_GPS_MAX_BUFFER_POINTS", "2000"))
CIRCUIT_OPEN_SEC = float(os.getenv("WS_GPS_CIRCUIT_OPEN_SEC", "60"))
MAX_BACKOFF_SEC = float(os.getenv("WS_GPS_MAX_BACKOFF_SEC", "30"))
_HTTP_TIMEOUT = httpx.Timeout(
    connect=float(os.getenv("WS_GPS_INGEST_CONNECT_TIMEOUT_SEC", "0.5")),
    read=float(os.getenv("WS_GPS_INGEST_READ_TIMEOUT_SEC", "5.0")),
    write=float(os.getenv("WS_GPS_INGEST_READ_TIMEOUT_SEC", "5.0")),
    pool=5.0,
)
# Marge d'horloge vs skew backend passé 24 h
PURGE_MAX_AGE = timedelta(hours=23, minutes=50)

_lock = threading.RLock()
# Buffer mémoire FIFO : (driver_id, point, enqueued_at_mono)
_buffer: deque[tuple[int, dict[str, Any], float]] = deque()
_last_flush_mono = 0.0
_request_times: dict[int, list[float]] = {}
_queue_depth = 0
_dropped_points = 0
_dropped_oldest = 0
_purged_skew = 0
_ingest_requests = 0
_retry_total = 0
_rejected_total = 0
_dlq_total = 0
_acked_total = 0
_spooled_total = 0

# Circuit breaker : closed | open | half_open
_circuit_state = "closed"
_circuit_opened_at = 0.0
_half_open_in_flight = False
_backoff_until = 0.0
_consecutive_failures = 0


def _service_token() -> str:
    return (os.getenv("INTERNAL_SERVICE_TOKEN") or "").strip()


def _is_redis_backend() -> bool:
    return SPOOL_BACKEND == "redis_stream"


def stats() -> dict[str, int | str | bool]:
    with _lock:
        depth = _queue_depth
        if _is_redis_backend():
            client = spool.get_redis()
            if client is not None:
                try:
                    depth = int(client.get(spool.STATS_PENDING_EVENTS) or 0)
                except Exception:
                    depth = _queue_depth
        return {
            "queue_depth": depth,
            "dropped_points": _dropped_points,
            "dropped_oldest": _dropped_oldest,
            "purged_skew": _purged_skew,
            "ingest_requests": _ingest_requests,
            "retry_total": _retry_total,
            "rejected_total": _rejected_total,
            "dlq_total": _dlq_total,
            "acked_total": _acked_total,
            "spooled_total": _spooled_total,
            "circuit_state": _circuit_state,
            "spool_backend": SPOOL_BACKEND,
            "flush_enabled": FLUSH_ENABLED,
        }


def _rate_ok(driver_id: int) -> bool:
    now = time.time()
    window = [t for t in _request_times.get(driver_id, []) if now - t < 60]
    _request_times[driver_id] = window
    return len(window) < RATE_LIMIT_PER_MIN


def _recompute_depth() -> None:
    global _queue_depth
    _queue_depth = len(_buffer)


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


def _extract_company_id(point: dict[str, Any]) -> int | None:
    raw = point.get("company_id")
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    return None


def enqueue_point(driver_id: int, point: dict[str, Any]) -> bool:
    """Ajoute un point. ``driver_id`` session socket uniquement. False si spool plein."""
    global _spooled_total, _rejected_total, _dropped_points, _queue_depth

    if not isinstance(point, dict):
        return False
    # Ne jamais faire confiance à un driver_id fourni par le mobile.
    safe_point = {k: v for k, v in point.items() if k != "driver_id"}
    company_id = _extract_company_id(safe_point)

    if _is_redis_backend():
        client = spool.get_redis()
        if client is None:
            with _lock:
                _rejected_total += 1
            return False
        try:
            phash, _ = eph.compute_event_payload_hash_from_point(safe_point)
        except Exception as exc:
            logger.warning(
                "gps enqueue hash failed driver_id=%s: %s",
                driver_id,
                type(exc).__name__,
            )
            with _lock:
                _rejected_total += 1
            return False
        ok, detail = spool.admit(
            client,
            driver_id=driver_id,
            company_id=company_id,
            point=safe_point,
            event_payload_hash=phash,
        )
        with _lock:
            if ok:
                _spooled_total += 1
                try:
                    _queue_depth = int(client.get(spool.STATS_PENDING_EVENTS) or 0)
                except Exception:
                    pass
            else:
                _rejected_total += 1
                if detail == "spool_full":
                    _dropped_points += 1
                logger.warning(
                    "gps spool refuse driver_id=%s reason=%s", driver_id, detail
                )
        return ok

    with _lock:
        if len(_buffer) >= MAX_BUFFER_POINTS:
            _rejected_total += 1
            _dropped_points += 1
            logger.warning(
                "gps buffer full refuse driver_id=%s depth=%s",
                driver_id,
                len(_buffer),
            )
            return False
        _buffer.append((driver_id, safe_point, time.monotonic()))
        _spooled_total += 1
        _recompute_depth()
        return True


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
        "gps ingest circuit OPEN (auth/mismatch) — retry probe in %ss",
        CIRCUIT_OPEN_SEC,
    )


def _close_circuit() -> None:
    global _circuit_state, _half_open_in_flight, _consecutive_failures
    _circuit_state = "closed"
    _half_open_in_flight = False
    _consecutive_failures = 0
    logger.info("gps ingest circuit CLOSED")


def _schedule_backoff() -> None:
    """Backoff exponentiel avec jitter (5xx / timeout / ambigu)."""
    global _backoff_until, _consecutive_failures
    _consecutive_failures += 1
    exp = min(MAX_BACKOFF_SEC, (2 ** min(_consecutive_failures, 5)))
    jitter = random.uniform(0.0, 1.0)
    delay = min(MAX_BACKOFF_SEC, exp + jitter)
    _backoff_until = time.monotonic() + delay


def _requeue_front(items: list[tuple[int, dict[str, Any], float]]) -> None:
    """Remet en tête sans drop oldest (F-02 : pas de perte silencieuse post-admission)."""
    for item in reversed(items):
        _buffer.appendleft(item)
    _recompute_depth()


def _take_batch() -> list[tuple[int, dict[str, Any], float]]:
    """Extrait jusqu'à MAX_POINTS_PER_BATCH points du même chauffeur (tête FIFO)."""
    _purge_skew_locked()
    if not _buffer:
        return []
    first_driver = _buffer[0][0]
    batch: list[tuple[int, dict[str, Any], float]] = []
    while (
        _buffer
        and len(batch) < MAX_POINTS_PER_BATCH
        and _buffer[0][0] == first_driver
    ):
        batch.append(_buffer.popleft())
    _recompute_depth()
    return batch


def _build_batch_id(
    *,
    driver_id: int,
    company_id: int | None,
    points: list[dict[str, Any]],
) -> str | None:
    if company_id is None:
        return None
    try:
        events: list[tuple[str, str]] = []
        for pt in points:
            phash, _ = eph.compute_event_payload_hash_from_point(pt)
            events.append((str(pt["location_event_id"]), phash))
        return eph.compute_batch_id(
            driver_id=driver_id,
            company_id=company_id,
            events=events,
        )
    except Exception as exc:
        logger.warning(
            "gps batch_id compute failed driver_id=%s: %s",
            driver_id,
            type(exc).__name__,
        )
        return None


def _parse_error_code(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    err = data.get("error") or data.get("error_code") or ""
    return str(err)


def _ack_contract_ok(
    body: Any,
    *,
    sent_batch_id: str | None,
    sent_count: int,
) -> bool:
    """HTTP 200 valide uniquement si durabilité PostgreSQL + comptes cohérents."""
    if not isinstance(body, dict):
        return False
    if body.get("durability") != "postgres_committed":
        return False
    resp_batch = body.get("batch_id")
    if sent_batch_id is not None and resp_batch != sent_batch_id:
        return False
    if not isinstance(resp_batch, str) or not resp_batch:
        return False
    received = body.get("received")
    persisted = body.get("persisted")
    duplicates = body.get("duplicates")
    if not isinstance(received, int) or isinstance(received, bool):
        return False
    if not isinstance(persisted, int) or isinstance(persisted, bool):
        return False
    if not isinstance(duplicates, int) or isinstance(duplicates, bool):
        return False
    if received != sent_count:
        return False
    if received != persisted + duplicates:
        return False
    return True


def _ingest_headers(token: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Internal-Service": os.getenv("INTERNAL_SERVICE_AUDIENCE", "ws-service"),
        "X-Internal-Token": token,
    }


def _build_post_body(
    *,
    driver_id: int,
    points: list[dict[str, Any]],
    company_id: int | None,
    batch_id: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"driver_id": driver_id, "points": points}
    if batch_id is not None:
        body["batch_id"] = batch_id
    if company_id is not None:
        body["company_id"] = company_id
    return body


async def _post_ingest(
    *,
    driver_id: int,
    points: list[dict[str, Any]],
    company_id: int | None,
    batch_id: str | None,
    token: str,
) -> httpx.Response:
    headers = _ingest_headers(token)
    body = _build_post_body(
        driver_id=driver_id,
        points=points,
        company_id=company_id,
        batch_id=batch_id,
    )
    async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
        return await client.post(INGEST_URL, json=body, headers=headers)


async def flush_once() -> None:
    """Flush un lot (tests / boucle). Backend mémoire ou Redis Streams."""
    if not FLUSH_ENABLED:
        return
    if _is_redis_backend():
        await _flush_redis_once()
    else:
        await _flush_memory_once()


async def _flush_memory_once() -> None:
    global _ingest_requests, _retry_total, _last_flush_mono
    global _half_open_in_flight, _acked_total, _dlq_total

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
    company_id = None
    for pt in points:
        cid = _extract_company_id(pt)
        if cid is not None:
            company_id = cid
            break
    batch_id = _build_batch_id(
        driver_id=driver_id, company_id=company_id, points=points
    )

    if not token:
        logger.error(
            "gps ingest aborted: INTERNAL_SERVICE_TOKEN missing (F-01 fail-closed)"
        )
        with _lock:
            _retry_total += 1
            _requeue_front(batch)
            if _circuit_state == "half_open":
                _half_open_in_flight = False
            _schedule_backoff()
        return

    try:
        resp = await _post_ingest(
            driver_id=driver_id,
            points=points,
            company_id=company_id,
            batch_id=batch_id,
            token=token,
        )
    except Exception:
        with _lock:
            _retry_total += 1
            _requeue_front(batch)
            _schedule_backoff()
            if _circuit_state == "half_open":
                _half_open_in_flight = False
        logger.exception("gps ingest failed driver_id=%s", driver_id)
        return

    with _lock:
        _request_times.setdefault(driver_id, []).append(time.time())
        _ingest_requests += 1
        _last_flush_mono = time.monotonic()
        status = resp.status_code

        if status == 200:
            try:
                body = resp.json()
            except Exception:
                body = None
            if _ack_contract_ok(
                body, sent_batch_id=batch_id, sent_count=len(points)
            ):
                _close_circuit()
                _acked_total += len(points)
                return
            # 200 incomplet / ambigu — requeue, pas d'ACK
            _retry_total += 1
            _requeue_front(batch)
            _schedule_backoff()
            if _circuit_state == "half_open":
                _half_open_in_flight = False
            logger.warning(
                "gps ingest 200 contract invalid driver_id=%s requeue",
                driver_id,
            )
            return

        if status == 409:
            # Conflit déterministe — DLQ mémoire, aucun retry
            _dlq_total += len(points)
            if _circuit_state == "half_open":
                _half_open_in_flight = False
            logger.critical(
                "gps ingest 409 → DLQ driver_id=%s count=%s body=%s",
                driver_id,
                len(points),
                (resp.text or "")[:200],
            )
            return

        if status in (401, 403):
            _retry_total += 1
            _requeue_front(batch)
            _open_circuit()
            logger.warning("gps ingest http %s driver_id=%s", status, driver_id)
            return

        if status == 400:
            err = _parse_error_code(resp)
            if err == "batch_id_mismatch" or "schema" in err:
                _retry_total += 1
                _requeue_front(batch)
                _open_circuit()
                logger.critical(
                    "gps ingest 400 %s → pending+circuit driver_id=%s",
                    err or "schema",
                    driver_id,
                )
                return
            # Validation GPS — DLQ (pas de retry)
            _dlq_total += len(points)
            if _circuit_state == "half_open":
                _half_open_in_flight = False
            logger.warning(
                "gps ingest 400 validation → DLQ driver_id=%s err=%s body=%s",
                driver_id,
                err,
                (resp.text or "")[:200],
            )
            return

        # 429 / 5xx / autre — requeue + backoff
        _retry_total += 1
        _requeue_front(batch)
        _schedule_backoff()
        if _circuit_state == "half_open":
            _half_open_in_flight = False
        logger.warning("gps ingest http %s driver_id=%s requeue", status, driver_id)


async def _flush_redis_once() -> None:
    global _ingest_requests, _retry_total, _last_flush_mono
    global _half_open_in_flight, _acked_total, _dlq_total

    client = spool.get_redis()
    if client is None:
        return

    with _lock:
        if time.monotonic() < _backoff_until:
            return
        if not _circuit_allows_request():
            return

    try:
        entries = spool.read_batch(client, count=MAX_POINTS_PER_BATCH)
    except Exception:
        logger.exception("gps spool read_batch failed")
        with _lock:
            _retry_total += 1
            _schedule_backoff()
            if _circuit_state == "half_open":
                _half_open_in_flight = False
        return

    if not entries:
        with _lock:
            if _circuit_state == "half_open":
                _half_open_in_flight = False
        return

    driver_id = int(entries[0][1].get("driver_id") or 0)
    same = [(sid, fields) for sid, fields in entries if int(fields.get("driver_id") or 0) == driver_id]
    if not same or driver_id <= 0:
        with _lock:
            if _circuit_state == "half_open":
                _half_open_in_flight = False
        return

    with _lock:
        if not _rate_ok(driver_id):
            if _circuit_state == "half_open":
                _half_open_in_flight = False
            return

    lock_token = uuid.uuid4().hex
    try:
        locked = spool.acquire_driver_lock(client, driver_id, lock_token)
    except Exception:
        logger.exception("gps acquire_driver_lock failed driver_id=%s", driver_id)
        with _lock:
            if _circuit_state == "half_open":
                _half_open_in_flight = False
        return
    if not locked:
        with _lock:
            if _circuit_state == "half_open":
                _half_open_in_flight = False
        return

    try:
        now = time.time()
        live: list[tuple[str, dict[str, Any]]] = []
        for sid, fields in same:
            deadline = float(fields.get("replay_deadline") or 0)
            if deadline and now > deadline:
                status_dlq, _detail = spool.transfer_dlq(
                    client, sid, reason="max_age_exceeded"
                )
                with _lock:
                    if status_dlq in ("ok", "already"):
                        _dlq_total += 1
                    else:
                        _retry_total += 1
                continue
            live.append((sid, fields))

        if not live:
            return

        points = [fields["point"] for _, fields in live]
        stream_ids = [sid for sid, _ in live]
        company_id = None
        for _, fields in live:
            cid = fields.get("company_id")
            if isinstance(cid, int) and not isinstance(cid, bool) and cid > 0:
                company_id = cid
                break
            if company_id is None:
                company_id = _extract_company_id(fields.get("point") or {})

        batch_id = _build_batch_id(
            driver_id=driver_id, company_id=company_id, points=points
        )
        token = _service_token()
        if not token:
            logger.error(
                "gps ingest aborted: INTERNAL_SERVICE_TOKEN missing (F-01 fail-closed)"
            )
            with _lock:
                _retry_total += 1
                _schedule_backoff()
                if _circuit_state == "half_open":
                    _half_open_in_flight = False
            return

        try:
            resp = await _post_ingest(
                driver_id=driver_id,
                points=points,
                company_id=company_id,
                batch_id=batch_id,
                token=token,
            )
        except Exception:
            with _lock:
                _retry_total += 1
                _schedule_backoff()
                if _circuit_state == "half_open":
                    _half_open_in_flight = False
            logger.exception("gps ingest failed driver_id=%s", driver_id)
            return

        with _lock:
            _request_times.setdefault(driver_id, []).append(time.time())
            _ingest_requests += 1
            _last_flush_mono = time.monotonic()

        status = resp.status_code

        if status == 200:
            try:
                body = resp.json()
            except Exception:
                body = None
            if _ack_contract_ok(
                body, sent_batch_id=batch_id, sent_count=len(points)
            ):
                spool.ack_batch(client, stream_ids)
                with _lock:
                    _close_circuit()
                    _acked_total += len(points)
                return
            with _lock:
                _retry_total += 1
                _schedule_backoff()
                if _circuit_state == "half_open":
                    _half_open_in_flight = False
            logger.warning(
                "gps ingest 200 contract invalid driver_id=%s keep pending",
                driver_id,
            )
            return

        if status == 409:
            err = _parse_error_code(resp)
            dlq_reason = (
                "batch_tenant_mismatch"
                if err == "tenant_mismatch"
                else "batch_payload_conflict"
            )
            # P0-1 : quarantaine prioritaire (force) — aucun retry auto
            for sid in stream_ids:
                st, _d = spool.transfer_dlq(
                    client, sid, reason=dlq_reason, force=True
                )
                with _lock:
                    if st in ("ok", "already"):
                        _dlq_total += 1
                    else:
                        _retry_total += 1
                        logger.critical(
                            "gps DLQ transfer failed sid=%s status=%s", sid, st
                        )
            with _lock:
                if _circuit_state == "half_open":
                    _half_open_in_flight = False
            logger.critical(
                "gps ingest 409 → DLQ lot entier driver_id=%s count=%s reason=%s",
                driver_id,
                len(stream_ids),
                dlq_reason,
            )
            return

        if status in (401, 403):
            with _lock:
                _retry_total += 1
                _open_circuit()
            logger.warning("gps ingest http %s driver_id=%s", status, driver_id)
            return

        if status == 400:
            err = _parse_error_code(resp)
            if err == "batch_id_mismatch" or "schema" in err:
                with _lock:
                    _retry_total += 1
                    _open_circuit()
                logger.critical(
                    "gps ingest 400 %s → pending+circuit driver_id=%s",
                    err or "schema",
                    driver_id,
                )
                return
            for sid in stream_ids:
                st, _d = spool.transfer_dlq(client, sid, reason="validation")
                with _lock:
                    if st in ("ok", "already"):
                        _dlq_total += 1
                    else:
                        _retry_total += 1
            with _lock:
                if _circuit_state == "half_open":
                    _half_open_in_flight = False
            logger.warning(
                "gps ingest 400 validation → DLQ driver_id=%s err=%s",
                driver_id,
                err,
            )
            return

        with _lock:
            _retry_total += 1
            _schedule_backoff()
            if _circuit_state == "half_open":
                _half_open_in_flight = False
        logger.warning("gps ingest http %s driver_id=%s keep pending", status, driver_id)
    finally:
        try:
            spool.release_driver_lock(client, driver_id, lock_token)
        except Exception:
            logger.warning(
                "gps release_driver_lock failed driver_id=%s", driver_id
            )


async def flush_driver(driver_id: int) -> None:
    """Compat : déclenche un flush (le batch est FIFO multi-chauffeurs)."""
    _ = driver_id
    await flush_once()


async def flush_loop() -> None:
    while True:
        await asyncio.sleep(FLUSH_INTERVAL_SEC)
        if not FLUSH_ENABLED:
            continue
        for _ in range(20):
            if _is_redis_backend():
                client = spool.get_redis()
                if client is None:
                    break
                try:
                    depth = int(client.get(spool.STATS_PENDING_EVENTS) or 0)
                except Exception:
                    depth = 1
                if depth <= 0:
                    break
            else:
                with _lock:
                    empty = len(_buffer) == 0
                if empty:
                    break
            await flush_once()
