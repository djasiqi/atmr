from __future__ import annotations

import os
import time
from typing import Any, cast

from ext import redis_client

SPAM_BURST_WINDOW_SECONDS = int(os.getenv("CONTACT_SPAM_BURST_WINDOW_SECONDS", "30"))
SPAM_BURST_THRESHOLD = int(os.getenv("CONTACT_SPAM_BURST_THRESHOLD", "3"))
CONTACT_RATE_LIMITS = {
    "support": {"minute": 8, "day": 40},
    "family": {"minute": 5, "day": 30},
    "demo": {"minute": 3, "day": 20},
    "institution": {"minute": 5, "day": 30},
    "transport": {"minute": 5, "day": 30},
    "billing": {"minute": 5, "day": 30},
}


def _counter_key(ip_hash: str, category: str, scope: str) -> str:
    return f"contact:rate:{scope}:{category}:{ip_hash}"


def _incr_counter(key: str, ttl_seconds: int) -> int:
    if not redis_client:
        return 0
    value = cast(int | str | bytes, redis_client.incr(key))
    count = int(value)
    if count == 1:
        redis_client.expire(key, ttl_seconds)
    return count


def is_silent_spam(payload: dict[str, Any]) -> bool:
    return bool((payload.get("website") or "").strip())


def hit_rate_limit(ip_hash: str, category: str) -> bool:
    limits = CONTACT_RATE_LIMITS.get(category, {"minute": 5, "day": 30})
    minute_key = _counter_key(ip_hash, category, "minute")
    day_key = _counter_key(ip_hash, category, "day")
    minute_count = _incr_counter(minute_key, 60)
    day_count = _incr_counter(day_key, 86400)
    if minute_count and minute_count > limits["minute"]:
        return True
    return bool(day_count and day_count > limits["day"])


def in_cooldown(ip_hash: str, category: str) -> bool:
    if not redis_client:
        return False
    burst_key = _counter_key(ip_hash, category, "burst")
    count = _incr_counter(burst_key, SPAM_BURST_WINDOW_SECONDS)
    if count and count > SPAM_BURST_THRESHOLD:
        cooldown_key = _counter_key(ip_hash, category, "cooldown")
        redis_client.set(
            cooldown_key, str(int(time.time())), ex=SPAM_BURST_WINDOW_SECONDS
        )
    cooldown_key = _counter_key(ip_hash, category, "cooldown")
    return bool(redis_client.get(cooldown_key))


def minimal_spam_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": payload.get("category"),
        "client_request_id": payload.get("client_request_id"),
        "email": payload.get("email"),
        "name": payload.get("name"),
        "message_len": len((payload.get("message") or "").strip()),
    }
