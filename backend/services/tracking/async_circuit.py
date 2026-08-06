"""Circuit breaker partagé pour l'ingest tracking async (Kafka).

Le consumer écrit un heartbeat Redis ; un évaluateur périodique (ou lecture
paresseuse avec throttle) publie l'état global du circuit. La route GPS ne
fait qu'un GET Redis — jamais Prometheus ni Kafka.

États : open | half_open | closed
Indéterminé / Redis KO → sync (conservateur).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
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
HEALTH_GATE_ENABLED = (
    os.getenv("TRACKING_ASYNC_HEALTH_GATE_ENABLED", "true").lower()
    in ("1", "true", "yes", "on")
)

_eval_lock = threading.Lock()
_last_eval_at = 0.0
_consecutive_fail = 0
_consecutive_ok = 0


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_consumer_heartbeat(
    *,
    last_persist_success_at: str | None = None,
    last_error_at: str | None = None,
    lag: int | None = None,
    instance_id: str | None = None,
) -> None:
    """Écrit / rafraîchit le heartbeat (appelé après chaque poll, même vide)."""
    if not redis_client:
        return
    now = _utcnow_iso()
    payload: dict[str, Any] = {
        "instance_id": instance_id or socket.gethostname(),
        "last_poll_at": now,
        "updated_at": now,
    }
    if last_persist_success_at:
        payload["last_persist_success_at"] = last_persist_success_at
    if last_error_at:
        payload["last_error_at"] = last_error_at
    if lag is not None:
        payload["lag"] = int(lag)
    try:
        existing_raw = redis_client.get(HEARTBEAT_KEY)
        if existing_raw:
            existing = json.loads(
                existing_raw.decode("utf-8")
                if isinstance(existing_raw, (bytes, bytearray))
                else str(existing_raw)
            )
            if isinstance(existing, dict):
                if "process_started_at" in existing:
                    payload["process_started_at"] = existing["process_started_at"]
                if last_persist_success_at is None and existing.get(
                    "last_persist_success_at"
                ):
                    payload["last_persist_success_at"] = existing[
                        "last_persist_success_at"
                    ]
                if last_error_at is None and existing.get("last_error_at"):
                    payload["last_error_at"] = existing["last_error_at"]
        if "process_started_at" not in payload:
            payload["process_started_at"] = now
        redis_client.setex(HEARTBEAT_KEY, HEARTBEAT_TTL_SEC, json.dumps(payload))
    except Exception:
        logger.debug("heartbeat write failed", exc_info=True)


def mark_consumer_persist_success() -> None:
    write_consumer_heartbeat(last_persist_success_at=_utcnow_iso())


def mark_consumer_error() -> None:
    write_consumer_heartbeat(last_error_at=_utcnow_iso())


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


def _heartbeat_healthy(hb: dict[str, Any] | None) -> tuple[bool, str, float | None, int]:
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
    """Évaluateur périodique — écrit tracking:consumer:ingest:circuit."""
    global _last_eval_at, _consecutive_fail, _consecutive_ok

    now = time.time()
    with _eval_lock:
        if not force and (now - _last_eval_at) < CIRCUIT_EVAL_INTERVAL_SEC:
            existing = _read_json(CIRCUIT_KEY)
            if existing:
                return existing
        _last_eval_at = now

        prev = _read_json(CIRCUIT_KEY) or {}
        prev_state = str(prev.get("state") or "open")
        opened_at = prev.get("opened_at")
        healthy_since = prev.get("healthy_since")

        hb = _read_json(HEARTBEAT_KEY)
        healthy, reason, hb_age, lag = _heartbeat_healthy(hb)

        if healthy:
            _consecutive_ok += 1
            _consecutive_fail = 0
        else:
            _consecutive_fail += 1
            _consecutive_ok = 0

        state = prev_state
        if not HEALTH_GATE_ENABLED:
            state = "closed"
            reason = "health_gate_disabled"
        elif prev_state == "closed":
            if _consecutive_fail >= CIRCUIT_FAIL_THRESHOLD:
                state = "open"
                opened_at = _utcnow_iso()
                healthy_since = None
        elif prev_state == "open":
            open_age = _parse_iso_age_seconds(opened_at) or 0.0
            if open_age >= CIRCUIT_OPEN_MIN_SEC and _consecutive_ok >= 1:
                state = "half_open"
            elif _consecutive_ok >= CIRCUIT_OK_THRESHOLD:
                state = "closed"
                healthy_since = _utcnow_iso()
        elif prev_state == "half_open":
            if _consecutive_fail >= 1:
                state = "open"
                opened_at = _utcnow_iso()
                healthy_since = None
            elif _consecutive_ok >= CIRCUIT_OK_THRESHOLD:
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
            "consecutive_fail": _consecutive_fail,
            "consecutive_ok": _consecutive_ok,
        }
        if redis_client:
            try:
                redis_client.setex(CIRCUIT_KEY, HEARTBEAT_TTL_SEC * 3, json.dumps(payload))
            except Exception:
                logger.debug("circuit store failed", exc_info=True)
        return payload


def get_circuit_state() -> dict[str, Any]:
    """Lecture circuit pour la route GPS (évalue si stale)."""
    if not redis_client:
        return {"state": "open", "reason": "redis_unavailable"}
    circuit = _read_json(CIRCUIT_KEY)
    eval_age = _parse_iso_age_seconds(circuit.get("evaluated_at") if circuit else None)
    if circuit is None or eval_age is None or eval_age > CIRCUIT_EVAL_INTERVAL_SEC * 2:
        return evaluate_and_store_circuit(force=True)
    return circuit


def should_use_async_ingest() -> bool:
    """True uniquement si le circuit est closed. Indéterminé → False (sync)."""
    if not HEALTH_GATE_ENABLED:
        return True
    if not TRACKING_ASYNC_FLAG():
        return False
    try:
        state = str(get_circuit_state().get("state") or "open")
        return state == "closed"
    except Exception:
        logger.warning("circuit read failed → sync", exc_info=True)
        return False


def TRACKING_ASYNC_FLAG() -> bool:
    return os.getenv("TRACKING_INGEST_ASYNC_ENABLED", "false").lower() == "true"
