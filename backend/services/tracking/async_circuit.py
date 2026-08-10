"""Circuit breaker partagé Redis pour l'ingest tracking async (Kafka).

Seul le consumer (évaluateur) écrit l'état. La route GPS ne fait qu'un GET Redis.
Compteurs consecutive_fail / consecutive_ok stockés dans Redis (multi-worker).

États : open | half_open | closed
Indéterminé / Redis KO → sync (conservateur).
"""

from __future__ import annotations

import json
import logging
import os
import socket
from datetime import UTC, datetime
from typing import Any

from ext import redis_client

logger = logging.getLogger(__name__)

HEARTBEAT_KEY = os.getenv(
    "TRACKING_CONSUMER_HEARTBEAT_KEY", "tracking:consumer:ingest:heartbeat"
)
CIRCUIT_KEY = os.getenv(
    "TRACKING_CONSUMER_CIRCUIT_KEY", "tracking:consumer:ingest:circuit"
)
HEARTBEAT_TTL_SEC = max(5, int(os.getenv("TRACKING_CONSUMER_HEARTBEAT_TTL_SEC", "20")))
HEARTBEAT_STALE_SEC = max(
    5, int(os.getenv("TRACKING_CONSUMER_HEARTBEAT_STALE_SEC", "20"))
)
CIRCUIT_EVAL_INTERVAL_SEC = max(
    1, int(os.getenv("TRACKING_ASYNC_CIRCUIT_EVAL_INTERVAL_SEC", "5"))
)
CIRCUIT_OPEN_MIN_SEC = max(
    5, int(os.getenv("TRACKING_ASYNC_CIRCUIT_OPEN_MIN_SEC", "30"))
)
CIRCUIT_FAIL_THRESHOLD = max(
    1, int(os.getenv("TRACKING_ASYNC_CIRCUIT_FAIL_THRESHOLD", "3"))
)
CIRCUIT_OK_THRESHOLD = max(
    1, int(os.getenv("TRACKING_ASYNC_CIRCUIT_OK_THRESHOLD", "3"))
)
LAG_THRESHOLD = max(0, int(os.getenv("TRACKING_ASYNC_CIRCUIT_LAG_THRESHOLD", "500")))
CIRCUIT_TTL_SEC = max(HEARTBEAT_TTL_SEC * 3, 60)
HEALTH_GATE_ENABLED = os.getenv(
    "TRACKING_ASYNC_HEALTH_GATE_ENABLED", "true"
).lower() in ("1", "true", "yes", "on")


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(key: str) -> dict[str, Any] | None:
    if not redis_client:
        return None
    try:
        raw = redis_client.get(key)
        if not raw:
            return None
        text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_json(key: str, payload: dict[str, Any], ttl: int) -> None:
    if not redis_client:
        return
    try:
        redis_client.setex(key, ttl, json.dumps(payload, default=str))
    except Exception:
        logger.debug("redis write failed key=%s", key, exc_info=True)


def _parse_iso_age_seconds(iso_ts: str | None) -> float | None:
    if not iso_ts:
        return None
    try:
        ts = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return max(0.0, (datetime.now(UTC) - ts).total_seconds())
    except Exception:
        return None


def write_consumer_heartbeat(
    *,
    last_persist_success_at: str | None = None,
    last_error_at: str | None = None,
    lag: int | None = None,
    instance_id: str | None = None,
) -> None:
    """Écrit / rafraîchit le heartbeat (après chaque poll, même vide)."""
    if not redis_client:
        return
    now = _utcnow_iso()
    existing = _read_json(HEARTBEAT_KEY) or {}
    payload: dict[str, Any] = {
        "instance_id": instance_id or socket.gethostname(),
        "process_started_at": existing.get("process_started_at") or now,
        "last_poll_at": now,
        "updated_at": now,
        "lag": int(lag) if lag is not None else int(existing.get("lag") or 0),
    }
    if last_persist_success_at:
        payload["last_persist_success_at"] = last_persist_success_at
    elif existing.get("last_persist_success_at"):
        payload["last_persist_success_at"] = existing["last_persist_success_at"]
    if last_error_at:
        payload["last_error_at"] = last_error_at
    elif existing.get("last_error_at"):
        payload["last_error_at"] = existing["last_error_at"]
    _write_json(HEARTBEAT_KEY, payload, HEARTBEAT_TTL_SEC)


def mark_consumer_persist_success(*, lag: int | None = None) -> None:
    write_consumer_heartbeat(last_persist_success_at=_utcnow_iso(), lag=lag)


def mark_consumer_error(*, lag: int | None = None) -> None:
    write_consumer_heartbeat(last_error_at=_utcnow_iso(), lag=lag)


def open_circuit_immediate(*, reason: str = "consumer_down") -> dict[str, Any]:
    """Ouvre le circuit immédiatement (shutdown / fail-stop). Ne pas attendre le TTL."""
    prev = _read_json(CIRCUIT_KEY) or {}
    payload = {
        "state": "open",
        "evaluated_at": _utcnow_iso(),
        "opened_at": _utcnow_iso(),
        "healthy_since": None,
        "reason": reason,
        "heartbeat_age_seconds": None,
        "lag": int(prev.get("lag") or 0),
        "consecutive_fail": max(
            CIRCUIT_FAIL_THRESHOLD, int(prev.get("consecutive_fail") or 0)
        ),
        "consecutive_ok": 0,
    }
    _write_json(CIRCUIT_KEY, payload, CIRCUIT_TTL_SEC)
    # Invalider le heartbeat pour forcer stale immédiat côté évaluateur
    if redis_client:
        try:
            redis_client.delete(HEARTBEAT_KEY)
        except Exception:
            logger.debug("heartbeat delete on open_circuit failed", exc_info=True)
    return payload


def _heartbeat_healthy(
    hb: dict[str, Any] | None,
) -> tuple[bool, str, float | None, int]:
    if not hb:
        return False, "heartbeat_absent", None, 0
    age = _parse_iso_age_seconds(hb.get("last_poll_at") or hb.get("updated_at"))
    lag = int(hb.get("lag") or 0)
    if age is None:
        return False, "heartbeat_unparseable", None, lag
    if age > HEARTBEAT_STALE_SEC:
        return False, "heartbeat_stale", age, lag
    err_age = _parse_iso_age_seconds(hb.get("last_error_at"))
    if err_age is not None and err_age < HEARTBEAT_STALE_SEC:
        return False, "recent_error", age, lag
    if lag > LAG_THRESHOLD:
        return False, "lag_high", age, lag
    return True, "ok", age, lag


def evaluate_and_store_circuit(*, force: bool = False) -> dict[str, Any]:
    """Évaluateur — appelé uniquement par le consumer (pas par la route GPS)."""
    if not redis_client:
        return {"state": "open", "reason": "redis_unavailable"}

    prev = _read_json(CIRCUIT_KEY) or {}
    eval_age = _parse_iso_age_seconds(prev.get("evaluated_at"))
    if (
        not force
        and prev
        and eval_age is not None
        and eval_age < CIRCUIT_EVAL_INTERVAL_SEC
    ):
        return prev

    prev_state = str(prev.get("state") or "open")
    opened_at = prev.get("opened_at")
    healthy_since = prev.get("healthy_since")
    consecutive_fail = int(prev.get("consecutive_fail") or 0)
    consecutive_ok = int(prev.get("consecutive_ok") or 0)

    hb = _read_json(HEARTBEAT_KEY)
    healthy, reason, hb_age, lag = _heartbeat_healthy(hb)

    if healthy:
        consecutive_ok += 1
        consecutive_fail = 0
    else:
        consecutive_fail += 1
        consecutive_ok = 0

    state = prev_state
    if not HEALTH_GATE_ENABLED:
        state = "closed"
        reason = "health_gate_disabled"
    elif prev_state == "closed":
        if consecutive_fail >= CIRCUIT_FAIL_THRESHOLD:
            state = "open"
            opened_at = _utcnow_iso()
            healthy_since = None
    elif prev_state == "open":
        # P0.2 : rester open pendant OPEN_MIN ; jamais sauter direct vers closed
        open_age = _parse_iso_age_seconds(opened_at) or 0.0
        if open_age < CIRCUIT_OPEN_MIN_SEC:
            state = "open"
        elif consecutive_ok >= 1:
            state = "half_open"
    elif prev_state == "half_open":
        if consecutive_fail >= 1:
            state = "open"
            opened_at = _utcnow_iso()
            healthy_since = None
        elif consecutive_ok >= CIRCUIT_OK_THRESHOLD:
            state = "closed"
            healthy_since = _utcnow_iso()
    else:
        state = "open" if not healthy else "closed"
        if state == "open":
            opened_at = opened_at or _utcnow_iso()

    payload = {
        "state": state,
        "evaluated_at": _utcnow_iso(),
        "opened_at": opened_at,
        "healthy_since": healthy_since,
        "reason": reason,
        "heartbeat_age_seconds": hb_age,
        "lag": lag,
        "consecutive_fail": consecutive_fail,
        "consecutive_ok": consecutive_ok,
    }
    _write_json(CIRCUIT_KEY, payload, CIRCUIT_TTL_SEC)
    return payload


def get_circuit_state() -> dict[str, Any]:
    """Lecture seule pour la route GPS — jamais d'évaluation ici."""
    if not redis_client:
        return {"state": "open", "reason": "redis_unavailable"}
    circuit = _read_json(CIRCUIT_KEY)
    if circuit is None:
        return {"state": "open", "reason": "circuit_absent"}
    return circuit


def should_use_async_ingest() -> bool:
    """True seulement si circuit closed ET heartbeat frais (lecture seule Redis).

    P0.2 : après kill -9 / OOM, un circuit ``closed`` stale ne doit pas envoyer
    de 202 vers un consumer mort — le heartbeat absent/stale force le sync.
    """
    if not HEALTH_GATE_ENABLED:
        return True
    if os.getenv("TRACKING_INGEST_ASYNC_ENABLED", "false").lower() != "true":
        return False
    try:
        state = str(get_circuit_state().get("state") or "open")
        if state != "closed":
            return False
        hb = _read_json(HEARTBEAT_KEY)
        healthy, _reason, _age, _lag = _heartbeat_healthy(hb)
        return bool(healthy)
    except Exception:
        logger.warning("circuit/heartbeat read failed → sync", exc_info=True)
        return False
