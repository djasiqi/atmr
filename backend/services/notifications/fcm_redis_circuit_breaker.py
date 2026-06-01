"""Circuit breaker FCM partagé via Redis (coordination inter-workers)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

FCM_CB_OPEN_KEY = "fcm:circuit_breaker:open"
FCM_CB_FAILURES_KEY = "fcm:circuit_breaker:failures"
FCM_CB_PROBE_KEY = "fcm:circuit_breaker:half_open_probe"

FCM_CIRCUIT_BREAKER_FAILURE_THRESHOLD = max(
    1, int(os.getenv("FCM_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5"))
)
FCM_CIRCUIT_BREAKER_OPEN_SECONDS = max(
    1, int(os.getenv("FCM_CIRCUIT_BREAKER_OPEN_SECONDS", "60"))
)
FCM_CIRCUIT_BREAKER_WINDOW_SECONDS = max(
    1, int(os.getenv("FCM_CIRCUIT_BREAKER_WINDOW_SECONDS", "60"))
)


def _get_redis() -> Any | None:
    try:
        from ext import redis_client

        return redis_client
    except Exception:
        return None


def _safe_int(val: Any) -> int:
    if val is None or val == 0:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def allow_fcm_request() -> bool:
    """True si l'appel FCM est autorisé. Fail-open si Redis indisponible."""
    redis_client = _get_redis()
    if not redis_client:
        return True

    try:
        if redis_client.get(FCM_CB_OPEN_KEY):
            return False

        # Half-open : une seule sonde simultanée entre workers
        probe_set = redis_client.set(
            FCM_CB_PROBE_KEY,
            str(time.time()),
            nx=True,
            ex=max(5, FCM_CIRCUIT_BREAKER_OPEN_SECONDS // 2),
        )
        if probe_set:
            return True

        # Autre worker sonde déjà — refuser le surplus en half-open
        failures = _safe_int(redis_client.get(FCM_CB_FAILURES_KEY))
        if failures >= FCM_CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            return False
        return True
    except Exception as exc:
        logger.warning("[fcm_cb] allow check failed (%s), fail-open", exc)
        return True


def record_fcm_success() -> None:
    redis_client = _get_redis()
    if not redis_client:
        return
    try:
        redis_client.delete(FCM_CB_FAILURES_KEY, FCM_CB_PROBE_KEY)
    except Exception:
        pass


def record_fcm_retryable_failure() -> None:
    redis_client = _get_redis()
    if not redis_client:
        return
    try:
        redis_client.delete(FCM_CB_PROBE_KEY)
        count = redis_client.incr(FCM_CB_FAILURES_KEY)
        if count == 1:
            redis_client.expire(FCM_CB_FAILURES_KEY, FCM_CIRCUIT_BREAKER_WINDOW_SECONDS)
        if count >= FCM_CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            redis_client.setex(
                FCM_CB_OPEN_KEY,
                FCM_CIRCUIT_BREAKER_OPEN_SECONDS,
                "1",
            )
            redis_client.delete(FCM_CB_FAILURES_KEY)
            logger.warning(
                "[fcm_cb] Circuit breaker OPEN for %ss after %s retryable failures",
                FCM_CIRCUIT_BREAKER_OPEN_SECONDS,
                count,
            )
            try:
                from services.notifications.metrics import record_circuit_breaker_opened

                record_circuit_breaker_opened("fcm")
            except Exception:
                pass
    except Exception as exc:
        logger.warning("[fcm_cb] record failure failed: %s", exc)


def record_fcm_non_retryable_failure() -> None:
    redis_client = _get_redis()
    if not redis_client:
        return
    try:
        redis_client.delete(FCM_CB_PROBE_KEY)
    except Exception:
        pass
