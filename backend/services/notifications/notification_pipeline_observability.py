"""Traçabilité end-to-end du pipeline notifications (logs + métriques métier)."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

PIPELINE_EVENTS = frozenset(
    {
        "notification_created",
        "notification_kafka_published",
        "notification_kafka_consumed",
        "notification_fcm_sent",
        "notification_fcm_failed",
        "notification_mobile_received",
        "notification_skipped",
        "notification_deduped",
        "notification_throttled",
    }
)

_STAGE_BY_EVENT = {
    "notification_created": "fanout",
    "notification_kafka_published": "kafka_producer",
    "notification_kafka_consumed": "kafka_consumer",
    "notification_fcm_sent": "fcm",
    "notification_fcm_failed": "fcm",
    "notification_mobile_received": "mobile",
    "notification_skipped": "policy",
    "notification_deduped": "dedup",
    "notification_throttled": "throttle",
}


def log_notification_pipeline_event(
    event: str,
    *,
    notification_id: str | int | None = None,
    booking_id: int | str | None = None,
    driver_id: int | str | None = None,
    pipeline_stage: str | None = None,
    notification_type: str | None = None,
    correlation_id: str | None = None,
    **extra: Any,
) -> None:
    """Émet un log JSON corrélable pour diagnostic bout-en-bout."""
    stage = pipeline_stage or _STAGE_BY_EVENT.get(event, "unknown")
    payload: dict[str, Any] = {
        "event": event,
        "pipeline_stage": stage,
    }
    if notification_id is not None:
        payload["notification_id"] = notification_id
    if booking_id is not None:
        payload["booking_id"] = booking_id
    if driver_id is not None:
        payload["driver_id"] = driver_id
    if notification_type is not None:
        payload["notification_type"] = notification_type
    if correlation_id is not None:
        payload["correlation_id"] = correlation_id
    if extra:
        payload.update(extra)

    logger.info("[notification_pipeline] %s", json.dumps(payload, default=str))

    _record_business_metric(event, notification_type=notification_type)


def _record_business_metric(event: str, *, notification_type: str | None) -> None:
    ntype = notification_type or "unknown"
    try:
        from services.notifications.metrics import (
            record_pipeline_notification_created,
            record_pipeline_notification_delivered,
            record_pipeline_notification_failed,
            record_pipeline_notification_sent,
        )

        if event == "notification_created":
            record_pipeline_notification_created(ntype)
        elif event in ("notification_fcm_sent", "notification_kafka_published"):
            record_pipeline_notification_sent(ntype)
        elif event == "notification_fcm_failed":
            record_pipeline_notification_failed(ntype)
        elif event == "notification_mobile_received":
            record_pipeline_notification_delivered(ntype)
    except Exception:
        pass


def build_idempotency_key(
    *,
    driver_id: int,
    notification_type: str,
    title: str,
    body: str,
    data: dict[str, Any] | None = None,
) -> str:
    """Clé stable pour éviter double envoi FCM après replay Kafka."""
    import hashlib

    data = data or {}
    parts = [
        str(driver_id),
        notification_type,
        str(data.get("booking_id") or ""),
        str(data.get("event_id") or data.get("trace_id") or ""),
        title,
        body,
    ]
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]
    return f"push:idempotency:{digest}"


def claim_idempotency_key(key: str, *, ttl_s: int = 3600) -> bool:
    """True si cette clé est nouvelle (envoi autorisé). Fail-open sans Redis."""
    try:
        from ext import redis_client

        if not redis_client:
            return True
        return bool(redis_client.set(key, "1", nx=True, ex=ttl_s))
    except Exception:
        return True


class NotificationE2ETimer:
    """Mesure la latence E2E notification (created -> fcm_sent)."""

    def __init__(self) -> None:
        self._started_at: float | None = None

    def start(self) -> None:
        self._started_at = time.perf_counter()

    def observe_if_started(self) -> None:
        if self._started_at is None:
            return
        elapsed = time.perf_counter() - self._started_at
        try:
            from services.notifications.metrics import observe_notification_e2e_latency

            observe_notification_e2e_latency(elapsed)
        except Exception:
            pass
