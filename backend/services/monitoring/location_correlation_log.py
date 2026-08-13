"""Logs structurés corrélation localisation (PR2). Contrôlé par ``LOCATION_CORRELATION_LOGS``."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_last_log_mono: dict[int, float] = {}
_THROTTLE_SEC = 1.0


def _mode() -> str:
    return os.getenv("LOCATION_CORRELATION_LOGS", "0").strip().lower()


def _enabled() -> bool:
    return _mode() in ("1", "true", "yes", "verbose")


def _verbose() -> bool:
    return _mode() == "verbose"


def _throttle_allows(driver_id: int) -> bool:
    if _verbose():
        return True
    now = time.monotonic()
    last = _last_log_mono.get(driver_id, 0.0)
    if now - last < _THROTTLE_SEC:
        return False
    _last_log_mono[driver_id] = now
    return True


def log_driver_location_processed(
    *,
    driver_id: int,
    company_id: int | None,
    transport: str,
    location_mode: str,
    accept_status: str,
    accept_reason: str,
    location_event_id: str | None,
    capture_id: str | None = None,
) -> None:
    """Une ligne JSON (sans lat/lon). Throttle 1/s/chauffeur sauf mode verbose."""
    if not _enabled() or not _throttle_allows(driver_id):
        return
    payload: dict[str, Any] = {
        "event": "driver_location_processed",
        "location_event_id": location_event_id or "",
        "capture_id": capture_id or "",
        "driver_id": driver_id,
        "company_id": company_id if company_id is not None else 0,
        "transport": transport,
        "location_mode": location_mode,
        "accept_status": accept_status,
        "accept_reason": accept_reason or "",
    }
    logger.info("%s", json.dumps(payload, ensure_ascii=False))
