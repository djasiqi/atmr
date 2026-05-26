"""Logs structurés pour le pipeline push chauffeur (corrélation incident)."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_DRIVER_PUSH_STAGES = frozenset(
    {
        "driver_push.publish",
        "driver_push.handler",
        "driver_push.notify",
        "driver_push.enqueue",
        "driver_push.task_start",
        "driver_push.task_done",
        "driver_push.dedup_skipped",
    }
)


def log_driver_push_stage(
    stage: str,
    *,
    event_id: str | None = None,
    correlation_id: str | None = None,
    booking_id: int | str | None = None,
    driver_id: int | str | None = None,
    notification_type: str | None = None,
    celery_task_id: str | None = None,
    **extra: Any,
) -> None:
    """Émet un log JSON grep-friendly pour localiser où la chaîne push s'arrête."""
    if stage not in _DRIVER_PUSH_STAGES:
        stage = f"driver_push.{stage.removeprefix('driver_push.')}"

    payload: dict[str, Any] = {"stage": stage}
    if event_id is not None:
        payload["event_id"] = event_id
    if correlation_id is not None:
        payload["correlation_id"] = correlation_id
    if booking_id is not None:
        payload["booking_id"] = booking_id
    if driver_id is not None:
        payload["driver_id"] = driver_id
    if notification_type is not None:
        payload["notification_type"] = notification_type
    if celery_task_id is not None:
        payload["celery_task_id"] = celery_task_id
    if extra:
        payload.update(extra)

    logger.info("[driver_push_pipeline] %s", json.dumps(payload, default=str))
