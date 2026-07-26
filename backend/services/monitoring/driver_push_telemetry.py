"""Télémétrie mobile — enregistrement push chauffeur (observabilité gate FCM)."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_METRICS_ENABLED = os.getenv(
    "DRIVER_PUSH_TELEMETRY_METRICS_ENABLED", "true"
).lower() not in (
    "0",
    "false",
    "no",
    "off",
)

_PUSH_TELEMETRY_TOTAL = None

try:
    from prometheus_client import Counter
except ImportError:
    Counter = None

if Counter is not None and _METRICS_ENABLED:
    _PUSH_TELEMETRY_TOTAL = Counter(
        "driver_push_telemetry_events_total",
        "Événements télémétrie enregistrement push mobile",
        ["event", "platform"],
    )

ALLOWED_PUSH_TELEMETRY_EVENTS = frozenset(
    {
        "driver_push.bridge_mounted",
        "driver_push.disclosure_blocked",
        "driver_push.permission_blocked",
        "driver_push.get_token_failed",
        "driver_push.token_acquired",
        "driver_push.register_success",
    }
)


def _safe_str(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def ingest_driver_push_telemetry(
    *, driver_id: int, body: dict[str, Any]
) -> dict[str, Any]:
    """Journalise un événement push mobile et incrémente Prometheus si disponible."""
    event = _safe_str(body.get("event"), default="")
    if event not in ALLOWED_PUSH_TELEMETRY_EVENTS:
        return {"ok": False, "error": "unknown_event", "event": event or None}

    platform = _safe_str(body.get("platform"))
    source = _safe_str(body.get("source"), default="mobile")
    stage = body.get("stage")
    reason = body.get("reason")
    provider = body.get("provider")
    enabled = body.get("enabled")
    fcm_enabled = body.get("fcm_enabled")
    context_type = body.get("context_type")
    error_code = body.get("error_code")
    token_length = body.get("token_length")

    logger.info(
        "driver_push_telemetry event=%s driver_id=%s platform=%s source=%s "
        "provider=%s stage=%s reason=%s enabled=%s fcm_enabled=%s context_type=%s "
        "error_code=%s token_length=%s",
        event,
        driver_id,
        platform,
        source,
        provider,
        stage,
        reason,
        enabled,
        fcm_enabled,
        context_type,
        error_code,
        token_length,
    )

    if _PUSH_TELEMETRY_TOTAL is not None:
        try:
            _PUSH_TELEMETRY_TOTAL.labels(event=event, platform=platform).inc()
        except Exception:
            logger.debug(
                "driver_push_telemetry prometheus increment failed", exc_info=True
            )

    return {"ok": True, "event": event}
