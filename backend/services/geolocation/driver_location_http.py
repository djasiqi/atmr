"""Rate limit et idempotence pour PUT /driver/me/location (HTTP fallback tracking).

Plus permissif que le socket (fenêtre large) mais borné ; fail-open si Redis indisponible.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from ext import redis_client

logger = logging.getLogger(__name__)

# Défaut : 60 requêtes / 60 s / chauffeur — au-dessus du plafond socket typique mais limité.
_DEFAULT_REQ = int(os.getenv("HTTP_DRIVER_LOCATION_REQ_PER_WINDOW", "60"))
_DEFAULT_WINDOW_SEC = int(os.getenv("HTTP_DRIVER_LOCATION_WINDOW_SEC", "60"))
_IDEMPOTENCY_TTL = int(os.getenv("HTTP_DRIVER_LOCATION_IDEMPOTENCY_TTL_SEC", "300"))


def check_http_driver_location_rate_limit(driver_id: int) -> tuple[bool, int | None]:
    """Retourne (allowed, retry_after_seconds). Fail-open si Redis absent ou erreur."""
    if not redis_client:
        return True, None
    limit = max(1, _DEFAULT_REQ)
    window_seconds = max(1, _DEFAULT_WINDOW_SEC)
    key = f"http_rate:driver_location:driver:{driver_id}"
    try:
        now = datetime.now(UTC)
        now_ts = int(now.timestamp())
        window_start_ts = now_ts - window_seconds
        pipe = redis_client.pipeline()
        pipe.zremrangebyscore(key, 0, window_start_ts)
        pipe.zcard(key)
        pipe.zadd(key, {str(now_ts): now_ts})
        pipe.expire(key, window_seconds + 1)
        results = pipe.execute()
        count = results[1] if results and len(results) > 1 else 0
        if count >= limit:
            oldest_ts_list = redis_client.zrange(key, 0, 0, withscores=True)
            if (
                oldest_ts_list
                and isinstance(oldest_ts_list, list)
                and len(oldest_ts_list) > 0
            ):
                oldest_tuple = oldest_ts_list[0]
                if isinstance(oldest_tuple, (tuple, list)) and len(oldest_tuple) > 1:
                    oldest = int(oldest_tuple[1])
                    retry_after = max(1, (oldest + window_seconds) - now_ts)
                else:
                    retry_after = window_seconds
            else:
                retry_after = window_seconds
            logger.warning(
                "HTTP driver location rate limit driver_id=%s count=%s limit=%s",
                driver_id,
                count + 1,
                limit,
            )
            return False, retry_after
        return True, None
    except Exception as e:
        logger.warning("HTTP driver location RL fail-open: %s", e)
        return True, None


def _idem_key(driver_id: int, idempotency_key: str) -> str:
    h = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:32]
    return f"driver_http_idem:{driver_id}:{h}"


def get_idempotent_response(
    driver_id: int, idempotency_key: str
) -> dict[str, Any] | None:
    """Si une réponse 200 a déjà été enregistrée pour cette clé, la retourne."""
    if not redis_client or not idempotency_key.strip():
        return None
    rkey = _idem_key(driver_id, idempotency_key.strip())
    try:
        raw = redis_client.get(rkey)
        if not raw:
            return None
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        return json.loads(text)
    except Exception as e:
        logger.debug("idem get failed: %s", e)
        return None


def store_idempotent_response(
    driver_id: int, idempotency_key: str, payload: dict[str, Any]
) -> None:
    if not redis_client or not idempotency_key.strip():
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
