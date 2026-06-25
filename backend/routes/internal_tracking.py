"""Ingestion GPS interne (ws-service → backend, pas d'écriture DB depuis ws-service)."""

from __future__ import annotations

import logging
import os

from flask import Blueprint, jsonify, request

internal_tracking_bp = Blueprint("internal_tracking", __name__)
logger = logging.getLogger(__name__)

_INTERNAL_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "")
_INTERNAL_INGEST_ENABLED = os.getenv(
    "INTERNAL_TRACKING_INGEST_ENABLED", "true"
).lower() not in ("0", "false", "no", "off")


def _authorized() -> bool:
    if not _INTERNAL_TOKEN:
        return True
    header = request.headers.get("X-Internal-Token", "")
    return header == _INTERNAL_TOKEN


@internal_tracking_bp.route("/api/internal/tracking/ingest", methods=["POST"])
def tracking_ingest():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401

    if not _INTERNAL_INGEST_ENABLED:
        return jsonify({"error": "disabled"}), 501

    data = request.get_json(silent=True) or {}
    driver_id = data.get("driver_id")
    points = data.get("points")
    if not isinstance(driver_id, int) or not isinstance(points, list):
        return jsonify({"error": "invalid_payload"}), 400

    from services.tracking import enqueue_tracking_event

    accepted = 0
    for raw in points:
        if not isinstance(raw, dict):
            continue
        lat = raw.get("latitude")
        lon = raw.get("longitude")
        if lat is None or lon is None:
            continue
        payload = {
            "latitude": float(lat),
            "longitude": float(lon),
            "accuracy": raw.get("accuracy"),
            "heading": raw.get("heading"),
            "speed": raw.get("speed"),
            "recorded_at": raw.get("recorded_at") or raw.get("timestamp"),
            "mission_id": raw.get("mission_id"),
            "location_mode": raw.get("location_mode") or "mission_live",
            "location_event_id": raw.get("location_event_id"),
        }
        result = enqueue_tracking_event(
            driver_id=driver_id,
            payload=payload,
            source="internal_http",
            company_id=raw.get("company_id")
            if isinstance(raw.get("company_id"), int)
            else None,
        )
        if result.get("queued") or result.get("trace_id"):
            accepted += 1

    logger.info(
        "[internal_tracking] ingest driver_id=%s accepted=%s total=%s",
        driver_id,
        accepted,
        len(points),
    )
    return jsonify({"ok": True, "accepted": accepted, "driver_id": driver_id}), 200
