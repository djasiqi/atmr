"""Télémétrie mobile — push entreprise (offres institution)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ALLOWED_COMPANY_PUSH_TELEMETRY_EVENTS = frozenset(
    {
        "company_push.new_request.opened",
        "company_push.new_request.tap_without_network",
        "company_push.new_request.open_to_accept",
    }
)


def _safe_str(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def ingest_company_push_telemetry(
    *, company_id: int, body: dict[str, Any]
) -> dict[str, Any]:
    """Journalise un événement push mobile entreprise et incrémente Prometheus."""
    event = _safe_str(body.get("event"), default="")
    if event not in ALLOWED_COMPANY_PUSH_TELEMETRY_EVENTS:
        return {"ok": False, "error": "unknown_event", "event": event or None}

    offer_id = body.get("offer_id")
    request_id = body.get("request_id")
    platform = _safe_str(body.get("platform"))
    source = _safe_str(body.get("source"), default="mobile")
    seconds = body.get("seconds")

    logger.info(
        "company_push_telemetry event=%s company_id=%s offer_id=%s request_id=%s "
        "platform=%s source=%s seconds=%s",
        event,
        company_id,
        offer_id,
        request_id,
        platform,
        source,
        seconds,
    )

    try:
        if event == "company_push.new_request.opened":
            from services.metrics.institution_metrics import (
                track_company_push_new_request_opened,
            )

            track_company_push_new_request_opened(company_id=company_id)
        elif event == "company_push.new_request.tap_without_network":
            from services.metrics.institution_metrics import (
                track_company_push_tap_without_network,
            )

            track_company_push_tap_without_network(company_id=company_id)
        elif event == "company_push.new_request.open_to_accept":
            if seconds is not None:
                from services.metrics.institution_metrics import (
                    track_company_push_open_to_accept_seconds,
                )

                track_company_push_open_to_accept_seconds(
                    seconds=float(seconds),
                )
    except Exception:
        logger.debug("company_push_telemetry metrics failed", exc_info=True)

    return {"ok": True, "event": event}
