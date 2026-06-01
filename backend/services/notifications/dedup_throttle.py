# backend/services/notifications/dedup_throttle.py
"""Dédoublonnage et throttle des push (Redis TTL)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)

# Préfixe des clés Redis pour push
PUSH_DEDUP_PREFIX = "push:dedup:"
PUSH_THROTTLE_PREFIX = "push:throttle:"

# INCR + EXPIRE atomique (évite clé sans TTL si expire échoue après incr)
_THROTTLE_LUA = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return count
"""


def _get_redis() -> Redis | None:
    try:
        from ext import redis_client

        return redis_client
    except Exception:
        return None


def should_skip_dedup(recipient_role: str, recipient_id: int, dedupe_key: str) -> bool:
    """True si cette push est un doublon (déjà envoyée récemment)."""
    redis_client = _get_redis()
    if not redis_client:
        return False
    key = f"{PUSH_DEDUP_PREFIX}{recipient_role}:{recipient_id}:{dedupe_key}"
    try:
        # SET NX EX atomique — évite la race GET puis SETEX sous charge multi-worker
        was_set = redis_client.set(key, "1", nx=True, ex=300)
        if not was_set:
            logger.info(
                "event=push_deduped recipient_role=%s recipient_id=%s dedupe_key=%s",
                recipient_role,
                recipient_id,
                dedupe_key,
            )
            try:
                from services.notifications.metrics import record_dedup_hit
                from services.notifications.notification_pipeline_observability import (
                    log_notification_pipeline_event,
                )

                record_dedup_hit()
                log_notification_pipeline_event(
                    "notification_deduped",
                    driver_id=recipient_id if recipient_role == "driver" else None,
                    pipeline_stage="dedup",
                    dedupe_key=dedupe_key,
                    recipient_role=recipient_role,
                    recipient_id=recipient_id,
                )
            except Exception:
                pass
            return True
        return False
    except Exception as e:
        logger.warning("dedup check failed: %s. Continuing.", e)
        return False


def should_skip_throttle(
    recipient_role: str,
    recipient_id: int,
    scope_key: str,
    window_s: int,
    max_per_window: int,
) -> bool:
    """True si le throttle est dépassé (trop de push dans la fenêtre)."""
    redis_client = _get_redis()
    if not redis_client or max_per_window <= 0:
        return False
    key = f"{PUSH_THROTTLE_PREFIX}{recipient_role}:{recipient_id}:{scope_key}"
    try:
        raw_count = redis_client.eval(_THROTTLE_LUA, 1, key, window_s)
        count = cast(int, raw_count) if raw_count is not None else 0
        if count > max_per_window:
            logger.info(
                "event=push_throttled recipient_role=%s recipient_id=%s scope=%s count=%s max=%s",
                recipient_role,
                recipient_id,
                scope_key,
                count,
                max_per_window,
            )
            try:
                from services.notifications.metrics import record_throttle_block
                from services.notifications.notification_pipeline_observability import (
                    log_notification_pipeline_event,
                )

                record_throttle_block()
                log_notification_pipeline_event(
                    "notification_throttled",
                    driver_id=recipient_id if recipient_role == "driver" else None,
                    pipeline_stage="throttle",
                    scope_key=scope_key,
                    count=count,
                    max_per_window=max_per_window,
                    recipient_role=recipient_role,
                    recipient_id=recipient_id,
                )
            except Exception:
                pass
            return True
        return False
    except Exception as e:
        logger.warning("throttle check failed: %s. Continuing.", e)
        return False


def check_dedup_and_throttle(
    recipient_role: str,
    recipient_id: int,
    dedupe_key: str,
    throttle_scope_key: str | None,
    throttle_window_s: int,
    throttle_max: int,
) -> tuple[bool, str | None]:
    """Vérifie dedup puis throttle. Retourne (skip, reason)."""
    if should_skip_dedup(recipient_role, recipient_id, dedupe_key):
        return True, "deduped"
    if (
        throttle_scope_key is not None
        and throttle_max > 0
        and should_skip_throttle(
            recipient_role,
            recipient_id,
            throttle_scope_key,
            throttle_window_s,
            throttle_max,
        )
    ):
        return True, "throttled"
    return False, None
