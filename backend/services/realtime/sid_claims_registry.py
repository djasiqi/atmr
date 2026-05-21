"""Registre distribué des claims JWT par SID Socket.IO (multi-worker Gunicorn)."""

from __future__ import annotations

import json
import logging
from contextlib import suppress
from typing import Any, cast

from ext import redis_client

logger = logging.getLogger(__name__)

SID_CLAIMS_PREFIX = "socketio:claims:"
SID_CLAIMS_TTL_SECONDS = 3600

# Fallback process-local when Redis is unavailable (dev only).
_LOCAL_SID_CLAIMS: dict[str, dict[str, Any]] = {}


def _claims_key(sid: str) -> str:
    return f"{SID_CLAIMS_PREFIX}{sid}"


def _resolve_redis_value(value: Any) -> Any:
    if hasattr(value, "__await__"):
        return None
    return value


def set_sid_claims(sid: str, data: dict[str, Any]) -> None:
    """Persiste les claims pour un SID (Redis + cache local)."""
    if not sid:
        return
    payload = dict(data)
    _LOCAL_SID_CLAIMS[sid] = payload
    if redis_client is None:
        return
    try:
        raw = json.dumps(payload)
        redis_client.setex(_claims_key(sid), SID_CLAIMS_TTL_SECONDS, raw)
    except Exception:
        logger.exception("[sid_claims] set failed sid=%s", sid)


def get_sid_claims(sid: str | None) -> dict[str, Any]:
    """Lit les claims : Redis d'abord, puis cache local."""
    if not sid:
        return {}
    if redis_client is not None:
        try:
            raw = _resolve_redis_value(cast(Any, redis_client.get(_claims_key(sid))))
            if isinstance(raw, (str, bytes, bytearray)):
                with suppress(json.JSONDecodeError):
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        _LOCAL_SID_CLAIMS[sid] = parsed
                        return parsed
        except Exception:
            logger.exception("[sid_claims] get failed sid=%s", sid)
    local = _LOCAL_SID_CLAIMS.get(sid)
    if isinstance(local, dict):
        return local
    return {}


def delete_sid_claims(sid: str | None) -> dict[str, Any] | None:
    """Supprime les claims ; retourne l'ancien payload local si présent."""
    if not sid:
        return None
    removed = _LOCAL_SID_CLAIMS.pop(sid, None)
    if redis_client is not None:
        with suppress(Exception):
            redis_client.delete(_claims_key(sid))
    return removed if isinstance(removed, dict) else None


def refresh_sid_claims_ttl(sid: str) -> None:
    """Prolonge le TTL Redis des claims (heartbeat optionnel)."""
    if not sid or redis_client is None:
        return
    with suppress(Exception):
        redis_client.expire(_claims_key(sid), SID_CLAIMS_TTL_SECONDS)
