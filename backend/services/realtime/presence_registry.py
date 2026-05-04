"""Presence registry distribue pour RealtimeGateway (Redis)."""

from __future__ import annotations

import json
import logging
import time
from contextlib import suppress
from typing import Any, cast

from ext import redis_client

logger = logging.getLogger(__name__)

PRESENCE_TTL_SECONDS = 90
SID_INDEX_PREFIX = "presence:sid:"
DRIVER_CANONICAL_PREFIX = "presence:driver:"
DRIVER_SESSIONS_PREFIX = "presence:driver_sids:"


def _presence_key(role: str, user_id: int, sid: str) -> str:
    return f"presence:{role}:{user_id}:{sid}"


def _sid_index_key(sid: str) -> str:
    return f"{SID_INDEX_PREFIX}{sid}"


def _driver_canonical_key(driver_id: int) -> str:
    return f"{DRIVER_CANONICAL_PREFIX}{driver_id}"


def _driver_sessions_key(driver_id: int) -> str:
    return f"{DRIVER_SESSIONS_PREFIX}{driver_id}"


def _resolve_redis_response(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return None
    return value


def register_presence(
    *,
    sid: str,
    user_id: int,
    role: str,
    company_id: int | None = None,
    driver_id: int | None = None,
) -> None:
    if redis_client is None:
        return
    payload: dict[str, Any] = {
        "sid": sid,
        "user_id": user_id,
        "role": role,
        "company_id": company_id,
        "driver_id": driver_id,
        "last_seen_ms": int(time.time() * 1000),
    }
    key = _presence_key(role, user_id, sid)
    try:
        raw_payload = json.dumps(payload)
        redis_client.setex(key, PRESENCE_TTL_SECONDS, raw_payload)
        redis_client.setex(_sid_index_key(sid), PRESENCE_TTL_SECONDS, raw_payload)

        if driver_id is None:
            return

        canonical_key = _driver_canonical_key(driver_id)
        sessions_key = _driver_sessions_key(driver_id)
        previous_raw = _resolve_redis_response(
            cast(Any, redis_client.get(canonical_key))
        )
        previous_sid: str | None = None
        if isinstance(previous_raw, (str, bytes, bytearray)):
            with suppress(json.JSONDecodeError):
                previous_payload = json.loads(previous_raw)
                if isinstance(previous_payload, dict):
                    prev_sid_obj = previous_payload.get("sid")
                    if isinstance(prev_sid_obj, str):
                        previous_sid = prev_sid_obj

        redis_client.setex(canonical_key, PRESENCE_TTL_SECONDS, raw_payload)
        redis_client.sadd(sessions_key, sid)
        redis_client.expire(sessions_key, PRESENCE_TTL_SECONDS)
        if previous_sid and previous_sid != sid:
            redis_client.delete(_sid_index_key(previous_sid))
            redis_client.srem(sessions_key, previous_sid)
            logger.info(
                "[presence] arbitration driver_id=%s old_sid=%s new_sid=%s",
                driver_id,
                previous_sid,
                sid,
            )
    except Exception:
        logger.exception("[presence] register failed")


def refresh_presence(*, sid: str, user_id: int, role: str) -> None:
    if redis_client is None:
        return
    key = _presence_key(role, user_id, sid)
    try:
        redis_client.expire(key, PRESENCE_TTL_SECONDS)
        sid_payload_raw = _resolve_redis_response(
            cast(Any, redis_client.get(_sid_index_key(sid)))
        )
        redis_client.expire(_sid_index_key(sid), PRESENCE_TTL_SECONDS)
        if isinstance(sid_payload_raw, (str, bytes, bytearray)):
            with suppress(json.JSONDecodeError):
                sid_payload = json.loads(sid_payload_raw)
                if isinstance(sid_payload, dict):
                    driver_id_obj = sid_payload.get("driver_id")
                    if isinstance(driver_id_obj, int):
                        redis_client.expire(
                            _driver_canonical_key(driver_id_obj), PRESENCE_TTL_SECONDS
                        )
                        redis_client.expire(
                            _driver_sessions_key(driver_id_obj), PRESENCE_TTL_SECONDS
                        )
    except Exception:
        logger.exception("[presence] refresh failed")


def remove_presence(*, sid: str, user_id: int, role: str) -> None:
    if redis_client is None:
        return
    key = _presence_key(role, user_id, sid)
    try:
        redis_client.delete(key)
        sid_payload_raw = _resolve_redis_response(
            cast(Any, redis_client.get(_sid_index_key(sid)))
        )
        redis_client.delete(_sid_index_key(sid))
        if isinstance(sid_payload_raw, (str, bytes, bytearray)):
            with suppress(json.JSONDecodeError):
                sid_payload = json.loads(sid_payload_raw)
                if isinstance(sid_payload, dict):
                    driver_id_obj = sid_payload.get("driver_id")
                    if isinstance(driver_id_obj, int):
                        sessions_key = _driver_sessions_key(driver_id_obj)
                        canonical_key = _driver_canonical_key(driver_id_obj)
                        redis_client.srem(sessions_key, sid)
                        remaining_sessions_raw = _resolve_redis_response(
                            cast(Any, redis_client.scard(sessions_key))
                        )
                        if int(remaining_sessions_raw or 0) <= 0:
                            redis_client.delete(sessions_key)
                        canonical_raw = _resolve_redis_response(
                            cast(Any, redis_client.get(canonical_key))
                        )
                        if isinstance(canonical_raw, (str, bytes, bytearray)):
                            with suppress(json.JSONDecodeError):
                                canonical_payload = json.loads(canonical_raw)
                                if (
                                    isinstance(canonical_payload, dict)
                                    and canonical_payload.get("sid") == sid
                                ):
                                    redis_client.delete(canonical_key)
    except Exception:
        logger.exception("[presence] remove failed")
