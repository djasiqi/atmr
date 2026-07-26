"""Garde body bornée pour POST /api/internal/tracking/ingest (avant précache JSON)."""

from __future__ import annotations

from io import BytesIO

from flask import Flask, jsonify, request
from werkzeug.wsgi import LimitedStream

INGEST_PATH = "/api/internal/tracking/ingest"
MAX_INGEST_BODY_BYTES = 64 * 1024


def register_internal_tracking_ingest_body_guard(app: Flask) -> None:
    """Enregistrer **avant** ``register_json_body_precache``.

    - Content-Length > 64 KiB → 413
    - Sans Content-Length / chunked → lecture bornée 64 KiB + 1
    - Réinjecte le buffer via BytesIO pour les lecteurs suivants (get_json)
    """

    @app.before_request
    def _guard_internal_tracking_ingest_body():  # pyright: ignore[reportUnusedFunction]
        if request.method != "POST":
            return None
        if request.path != INGEST_PATH:
            return None

        content_length = request.content_length
        if content_length is not None and content_length > MAX_INGEST_BODY_BYTES:
            return jsonify(
                {
                    "error": "payload_too_large",
                    "max_bytes": MAX_INGEST_BODY_BYTES,
                }
            ), 413

        # Lecture bornée (API publique LimitedStream) — pas de get_data non borné.
        wsgi_input = request.environ.get("wsgi.input")
        if wsgi_input is None:
            return None

        limited = LimitedStream(wsgi_input, MAX_INGEST_BODY_BYTES + 1)
        buf = limited.read(MAX_INGEST_BODY_BYTES + 1)
        if len(buf) > MAX_INGEST_BODY_BYTES:
            return jsonify(
                {
                    "error": "payload_too_large",
                    "max_bytes": MAX_INGEST_BODY_BYTES,
                }
            ), 413

        request.environ["wsgi.input"] = BytesIO(buf)
        request.environ["CONTENT_LENGTH"] = str(len(buf))
        # Cache pour get_json / get_data sans relecture non bornée.
        request.get_data(cache=True)
        return None
