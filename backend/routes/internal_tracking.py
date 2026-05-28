"""Ingestion GPS interne (ws-service → backend, pas d'écriture DB depuis ws-service)."""

from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

internal_tracking_bp = Blueprint("internal_tracking", __name__)

_INTERNAL_TOKEN = os.getenv("INTERNAL_SERVICE_TOKEN", "")


def _authorized() -> bool:
    if not _INTERNAL_TOKEN:
        return True
    header = request.headers.get("X-Internal-Token", "")
    return header == _INTERNAL_TOKEN


@internal_tracking_bp.route("/api/internal/tracking/ingest", methods=["POST"])
def tracking_ingest():
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    driver_id = data.get("driver_id")
    points = data.get("points")
    if not isinstance(driver_id, int) or not isinstance(points, list):
        return jsonify({"error": "invalid_payload"}), 400

    accepted = len(points)
    # Extension : brancher sur le pipeline tracking existant (chat.handle_driver_location_batch).
    return jsonify({"ok": True, "accepted": accepted, "driver_id": driver_id}), 200
