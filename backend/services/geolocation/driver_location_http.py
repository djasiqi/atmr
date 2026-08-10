"""Rate limit et idempotence pour PUT /driver/me/location (HTTP fallback tracking).

Limiteur atomique dual-fenêtre (Lua Redis) par chauffeur :
- short : 30 / 10 s (burst)
- long  : 120 / 60 s (soutenu)

Fail-soft si Redis indisponible : limiteur mémoire local borné (~300/min/worker).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from typing import Any

from ext import redis_client

logger = logging.getLogger(__name__)

# Dual-window (plan P0 DEPLOY-A)
_SHORT_LIMIT = max(1, int(os.getenv("HTTP_DRIVER_LOCATION_SHORT_LIMIT", "30")))
_SHORT_WINDOW_SEC = max(
    1, int(os.getenv("HTTP_DRIVER_LOCATION_SHORT_WINDOW_SEC", "10"))
)
_LONG_LIMIT = max(1, int(os.getenv("HTTP_DRIVER_LOCATION_LONG_LIMIT", "120")))
_LONG_WINDOW_SEC = max(1, int(os.getenv("HTTP_DRIVER_LOCATION_LONG_WINDOW_SEC", "60")))
# Compat anciens noms d'env → fenêtre longue si fournis
if os.getenv("HTTP_DRIVER_LOCATION_REQ_PER_WINDOW"):
    _LONG_LIMIT = max(1, int(os.getenv("HTTP_DRIVER_LOCATION_REQ_PER_WINDOW", "120")))
if os.getenv("HTTP_DRIVER_LOCATION_WINDOW_SEC"):
    _LONG_WINDOW_SEC = max(1, int(os.getenv("HTTP_DRIVER_LOCATION_WINDOW_SEC", "60")))

_IDEMPOTENCY_TTL = int(os.getenv("HTTP_DRIVER_LOCATION_IDEMPOTENCY_TTL_SEC", "300"))
_MEMORY_FALLBACK_LIMIT = max(
    1, int(os.getenv("HTTP_DRIVER_LOCATION_MEMORY_FALLBACK_LIMIT", "300"))
)
_MEMORY_FALLBACK_WINDOW_SEC = max(
    1, int(os.getenv("HTTP_DRIVER_LOCATION_MEMORY_FALLBACK_WINDOW_SEC", "60"))
)
_KEY_PREFIX = os.getenv(
    "HTTP_DRIVER_LOCATION_RATE_KEY_PREFIX", "http_rate:v2:driver_location"
)

# Lua : deux ZSET, une décision, ZADD uniquement si autorisé (count >= limit → reject)
_DUAL_WINDOW_LUA = """
local short_key = KEYS[1]
local long_key = KEYS[2]
local now = tonumber(ARGV[1])
local short_window = tonumber(ARGV[2])
local short_limit = tonumber(ARGV[3])
local long_window = tonumber(ARGV[4])
local long_limit = tonumber(ARGV[5])
local member = ARGV[6]

redis.call("ZREMRANGEBYSCORE", short_key, 0, now - short_window)
redis.call("ZREMRANGEBYSCORE", long_key, 0, now - long_window)

local short_count = redis.call("ZCARD", short_key)
local long_count = redis.call("ZCARD", long_key)

local short_blocked = short_count >= short_limit
local long_blocked = long_count >= long_limit

if short_blocked or long_blocked then
  local retry_after = 1
  local reason = "both"
  if short_blocked and not long_blocked then
    reason = "short_window"
  elseif long_blocked and not short_blocked then
    reason = "long_window"
  end
  local function window_retry(key, window)
    local oldest = redis.call("ZRANGE", key, 0, 0, "WITHSCORES")
    if oldest and #oldest >= 2 then
      return math.max(1, math.floor(tonumber(oldest[2]) + window - now))
    end
    return window
  end
  if short_blocked then
    retry_after = math.max(retry_after, window_retry(short_key, short_window))
  end
  if long_blocked then
    retry_after = math.max(retry_after, window_retry(long_key, long_window))
  end
  return {0, retry_after, short_count, long_count, reason}
end

redis.call("ZADD", short_key, now, member)
redis.call("ZADD", long_key, now, member)
redis.call("EXPIRE", short_key, short_window + 1)
redis.call("EXPIRE", long_key, long_window + 1)
return {1, 0, short_count + 1, long_count + 1, "ok"}
"""

_memory_lock = threading.Lock()
_memory_hits: dict[int, list[float]] = defaultdict(list)


def _inc_fallback_metric() -> None:
    try:
        from services.monitoring.driver_location_metrics import (
            inc_tracking_rate_limit_fallback,
        )

        inc_tracking_rate_limit_fallback()
    except Exception:
        logger.debug("rate_limit_fallback metric unavailable", exc_info=True)


def _check_memory_fallback(driver_id: int) -> tuple[bool, int | None, str | None]:
    """Limiteur mémoire local borné quand Redis est indisponible."""
    _inc_fallback_metric()
    now = time.time()
    window = float(_MEMORY_FALLBACK_WINDOW_SEC)
    limit = _MEMORY_FALLBACK_LIMIT
    with _memory_lock:
        hits = _memory_hits.setdefault(driver_id, [])
        cutoff = now - window
        hits[:] = [t for t in hits if t > cutoff]
        if len(hits) >= limit:
            oldest = hits[0] if hits else now
            retry_after = max(1, int(oldest + window - now) + 1)
            return False, retry_after, "memory_fallback"
        hits.append(now)
        return True, None, None


def check_http_driver_location_rate_limit(
    driver_id: int,
) -> tuple[bool, int | None, str | None]:
    """Retourne (allowed, retry_after_seconds, reason).

    reason ∈ {None, short_window, long_window, both, memory_fallback}.
    """
    if not redis_client:
        return _check_memory_fallback(driver_id)

    short_key = f"{_KEY_PREFIX}:short:driver:{driver_id}"
    long_key = f"{_KEY_PREFIX}:long:driver:{driver_id}"
    now_ms = time.time()
    member = f"{int(now_ms * 1000)}:{uuid.uuid4().hex[:12]}"

    try:
        result = redis_client.eval(
            _DUAL_WINDOW_LUA,
            2,
            short_key,
            long_key,
            now_ms,
            _SHORT_WINDOW_SEC,
            _SHORT_LIMIT,
            _LONG_WINDOW_SEC,
            _LONG_LIMIT,
            member,
        )
        if not result or len(result) < 2:
            logger.warning("HTTP driver location RL unexpected lua result: %s", result)
            return _check_memory_fallback(driver_id)

        allowed = int(result[0]) == 1
        retry_after = int(result[1]) if int(result[1]) > 0 else None
        reason = None
        if len(result) >= 5 and result[4] not in (None, b"ok", "ok"):
            raw_reason = result[4]
            reason = (
                raw_reason.decode("utf-8")
                if isinstance(raw_reason, (bytes, bytearray))
                else str(raw_reason)
            )
        if not allowed:
            logger.warning(
                "HTTP driver location rate limit driver_id=%s reason=%s "
                "short=%s/%s long=%s/%s retry_after=%s",
                driver_id,
                reason,
                result[2] if len(result) > 2 else "?",
                _SHORT_LIMIT,
                result[3] if len(result) > 3 else "?",
                _LONG_LIMIT,
                retry_after,
            )
            return False, retry_after or 1, reason
        return True, None, None
    except Exception as e:
        logger.warning("HTTP driver location RL redis error → memory fallback: %s", e)
        return _check_memory_fallback(driver_id)


def _idem_key(driver_id: int, idempotency_key: str) -> str:
    h = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:32]
    return f"driver_http_idem:{driver_id}:{h}"


def get_idempotent_response(
    driver_id: int, idempotency_key: str
) -> dict[str, Any] | None:
    """Si une réponse durable a déjà été enregistrée pour cette clé, la retourne."""
    if not redis_client or not idempotency_key.strip():
        return None
    rkey = _idem_key(driver_id, idempotency_key.strip())
    try:
        raw = redis_client.get(rkey)
        if not raw:
            return None
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        payload = json.loads(text)
        # Ne jamais rejouer un 202 queued_async comme réponse finale
        if isinstance(payload, dict):
            durability = str(payload.get("durability") or "")
            if durability == "queued_async" or payload.get("queued") is True:
                return None
            if durability and durability != "persisted_sync":
                if payload.get("ack_status") != "persisted":
                    return None
        return payload
    except Exception as e:
        logger.debug("idem get failed: %s", e)
        return None


def store_idempotent_response(
    driver_id: int, idempotency_key: str, payload: dict[str, Any]
) -> None:
    """Persiste uniquement les réponses durables (persisted_sync)."""
    if not redis_client or not idempotency_key.strip():
        return
    durability = str(payload.get("durability") or "")
    if durability == "queued_async" or payload.get("queued") is True:
        return
    if durability != "persisted_sync" and payload.get("ack_status") != "persisted":
        return
    rkey = _idem_key(driver_id, idempotency_key.strip())
    try:
        redis_client.setex(
            rkey,
            _IDEMPOTENCY_TTL,
            json.dumps(payload, default=str),
        )
    except Exception as e:
        logger.warning("idem store failed: %s", e)
