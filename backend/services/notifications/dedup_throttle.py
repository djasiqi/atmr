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
        if redis_client.get(key):
            logger.info(
                "event=push_deduped recipient_role=%s recipient_id=%s dedupe_key=%s",
                recipient_role,
                recipient_id,
                dedupe_key,
            )
            return True
        redis_client.setex(key, 300, "1")  # TTL 5 min
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
        raw_count = redis_client.incr(key)
        count = cast(int, raw_count) if raw_count is not None else 0
        if count == 1:
            redis_client.expire(key, window_s)
        if count > max_per_window:
            logger.info(
                "event=push_throttled recipient_role=%s recipient_id=%s scope=%s count=%s max=%s",
                recipient_role,
                recipient_id,
                scope_key,
                count,
                max_per_window,
            )
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
