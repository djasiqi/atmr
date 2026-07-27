"""Idempotence Redis pending→done (accélérateur F-01/F-02).

PostgreSQL (ledger) est l'autorité. Redis ne décide jamais seul un HTTP 200.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from typing import Any, Literal

from services.security.internal_service_auth import get_idempotency_ttls

logger = logging.getLogger(__name__)

# F-02 : plus de retry_later bloquant — pending = accélérateur seulement
IdempotencyOutcome = Literal[
    "reserved",
    "duplicate_hint",
    "pending_hint",
    "unavailable",
]

_LUA_MARK_DONE = """
local current = redis.call("GET", KEYS[1])
if current == ARGV[1] then
  redis.call("SET", KEYS[1], "done", "EX", tonumber(ARGV[2]))
  return 1
end
return 0
"""

_LUA_RELEASE_PENDING = """
local current = redis.call("GET", KEYS[1])
if current == ARGV[1] then
  redis.call("DEL", KEYS[1])
  return 1
end
return 0
"""


def redis_key_for_event(*, driver_id: int, location_event_id: str) -> str:
    digest = hashlib.sha256(location_event_id.encode("utf-8")).hexdigest()
    return f"tracking:ingest:{driver_id}:{digest}"


def _get_redis() -> Any | None:
    try:
        from ext import redis_client

        if redis_client is None:
            return None
        redis_client.ping()
        return redis_client
    except Exception:
        return None


def peek_idempotency_state(
    *, driver_id: int, location_event_id: str
) -> Literal["done", "pending", "absent", "unavailable"]:
    client = _get_redis()
    if client is None:
        return "unavailable"
    key = redis_key_for_event(driver_id=driver_id, location_event_id=location_event_id)
    try:
        existing = client.get(key)
        if existing is None:
            return "absent"
        if isinstance(existing, bytes):
            existing_s = existing.decode("utf-8", errors="replace")
        else:
            existing_s = str(existing)
        if existing_s == "done":
            return "done"
        if existing_s.startswith("pending:"):
            return "pending"
        return "absent"
    except Exception:
        return "unavailable"


def try_reserve(
    *,
    driver_id: int,
    location_event_id: str,
) -> tuple[IdempotencyOutcome, str | None]:
    """Réserve pending:{nonce} avec SET NX EX (accélérateur).

    F-02 : ``unavailable`` / ``pending_hint`` / ``duplicate_hint`` n'imposent
    pas de HTTP 503 — la route continue vers PostgreSQL.
    """
    client = _get_redis()
    if client is None:
        return "unavailable", None

    pending_ttl, _done_ttl = get_idempotency_ttls()
    key = redis_key_for_event(driver_id=driver_id, location_event_id=location_event_id)
    nonce = secrets.token_hex(16)
    pending_value = f"pending:{nonce}"

    try:
        ok = client.set(key, pending_value, nx=True, ex=pending_ttl)
        if ok:
            return "reserved", nonce
        existing = client.get(key)
        if existing is None:
            return "pending_hint", None
        if isinstance(existing, bytes):
            existing_s = existing.decode("utf-8", errors="replace")
        else:
            existing_s = str(existing)
        if existing_s == "done":
            return "duplicate_hint", None
        if existing_s.startswith("pending:"):
            return "pending_hint", None
        logger.warning(
            "[ingest_idempotency] valeur Redis inattendue key=%s value=%s",
            key,
            existing_s[:64],
        )
        return "pending_hint", None
    except Exception as exc:
        logger.warning("[ingest_idempotency] reserve failed: %s", type(exc).__name__)
        return "unavailable", None


def mark_done(*, driver_id: int, location_event_id: str, nonce: str | None) -> bool:
    """Best-effort post-commit PostgreSQL.

    - nonce fourni : Lua compare-and-set pending→done uniquement
    - nonce absent : SET done (accélérateur, pas d'autorité)
    """
    client = _get_redis()
    if client is None:
        return False
    _pending_ttl, done_ttl = get_idempotency_ttls()
    key = redis_key_for_event(driver_id=driver_id, location_event_id=location_event_id)
    try:
        if nonce:
            expected = f"pending:{nonce}"
            result = client.eval(_LUA_MARK_DONE, 1, key, expected, str(done_ttl))
            return int(result or 0) == 1
        client.set(key, "done", ex=done_ttl)
        return True
    except Exception as exc:
        logger.warning("[ingest_idempotency] mark_done failed: %s", type(exc).__name__)
        return False


def release_pending(*, driver_id: int, location_event_id: str, nonce: str) -> bool:
    client = _get_redis()
    if client is None:
        return False
    key = redis_key_for_event(driver_id=driver_id, location_event_id=location_event_id)
    expected = f"pending:{nonce}"
    try:
        result = client.eval(_LUA_RELEASE_PENDING, 1, key, expected)
        return int(result or 0) == 1
    except Exception as exc:
        logger.warning(
            "[ingest_idempotency] release_pending failed: %s", type(exc).__name__
        )
        return False


def new_trace_id() -> str:
    return str(uuid.uuid4())
