"""Publication best-effort vers ws-service via Redis pub/sub (canary / PR D)."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

from services.infrastructure.runtime_flags import is_ws_relay_publish_enabled

logger = logging.getLogger("ws_relay")

_RELAY_CHANNEL = os.getenv("WS_RELAY_CHANNEL", "ws:relay:events")
_TIMEOUT_MS = float(os.getenv("WS_RELAY_PUBLISH_TIMEOUT_MS", "80"))
_FAILURE_THRESHOLD = int(os.getenv("WS_RELAY_CIRCUIT_FAILURE_THRESHOLD", "10"))
_COOLDOWN_SEC = float(os.getenv("WS_RELAY_CIRCUIT_COOLDOWN_SEC", "60"))
_QUEUE_MAX = int(os.getenv("WS_RELAY_QUEUE_MAX_SIZE", "1000"))

_lock = threading.Lock()
_consecutive_failures = 0
_circuit_open_until = 0.0
_dropped_total = 0
_published_total = 0


def relay_stats() -> dict[str, int]:
    with _lock:
        return {
            "published": _published_total,
            "dropped": _dropped_total,
            "circuit_open": int(time.time() < _circuit_open_until),
        }


def _circuit_allows_publish() -> bool:
    global _circuit_open_until
    if time.time() < _circuit_open_until:
        return False
    return True


def _record_failure() -> None:
    global _consecutive_failures, _circuit_open_until
    with _lock:
        _consecutive_failures += 1
        if _consecutive_failures >= _FAILURE_THRESHOLD:
            _circuit_open_until = time.time() + _COOLDOWN_SEC
            logger.warning(
                "ws relay circuit open for %.0fs after %s failures",
                _COOLDOWN_SEC,
                _consecutive_failures,
            )


def _record_success() -> None:
    global _consecutive_failures
    with _lock:
        _consecutive_failures = 0


def _record_drop() -> None:
    global _dropped_total
    with _lock:
        _dropped_total += 1


def _record_published() -> None:
    global _published_total
    with _lock:
        _published_total += 1


def publish_relay_event(
    *,
    room: str,
    event_type: str,
    payload: dict[str, Any],
    criticality: str = "normal",
) -> None:
    """Fire-and-forget : ne lève jamais vers l'appelant."""
    if not is_ws_relay_publish_enabled():
        return
    if not _circuit_allows_publish():
        _record_drop()
        return

    try:
        from ext import redis_client
    except Exception:
        redis_client = None

    if redis_client is None:
        _record_failure()
        _record_drop()
        return

    body = json.dumps(
        {
            "room": room,
            "event_type": event_type,
            "payload": payload,
            "criticality": criticality,
            "ts": int(time.time() * 1000),
        },
        default=str,
    )

    def _do_publish() -> None:
        try:
            redis_client.publish(_RELAY_CHANNEL, body)  # type: ignore[union-attr]
            _record_success()
            _record_published()
        except Exception:
            logger.exception(
                "ws relay publish failed event=%s room=%s criticality=%s",
                event_type,
                room,
                criticality,
            )
            _record_failure()
            _record_drop()

    # Exécution synchrone courte (timeout implicite via socket_timeout Redis ~5s).
    # Pour respecter <100ms en charge, on pourrait passer à un thread pool dédié.
    deadline = time.monotonic() + (_TIMEOUT_MS / 1000.0)
    try:
        _do_publish()
    except Exception:
        pass
    if time.monotonic() > deadline:
        logger.warning("ws relay publish exceeded budget event=%s", event_type)
