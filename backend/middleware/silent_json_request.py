"""Requêtes JSON tolérantes : évite les 400 Werkzeug ``invalid_json`` prématurés."""

from __future__ import annotations

import json
from typing import Any

from flask import Flask, Request, request


class SilentJSONRequest(Request):
    """``get_json(silent=True)`` par défaut pour ne pas lever BadRequest en amont."""

    def get_json(
        self,
        force: bool = False,
        silent: bool = True,
        cache: bool = True,
    ) -> Any:
        return super().get_json(force=force, silent=silent, cache=cache)


def register_json_body_precache(app: Flask) -> None:
    """Met en cache le corps brut avant tout middleware qui le consommerait.

    F-01 : ignore ``/api/internal/tracking/ingest`` (garde bornée dédiée en amont).
    """

    @app.before_request
    def _precache_json_request_body():  # pyright: ignore[reportUnusedFunction]
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return
        if request.path == "/api/internal/tracking/ingest":
            return
        content_type = (request.content_type or "").lower()
        if "json" not in content_type:
            return
        request.get_data(cache=True)
        return


def redact_json_body_preview(raw: str | None, limit: int = 400) -> str:
    """Aperçu JSON pour logs (masque les champs sensibles)."""
    if not raw:
        return ""
    text = raw
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            redacted = {
                key: (
                    "***"
                    if any(
                        token in str(key).lower()
                        for token in ("password", "secret", "token")
                    )
                    else value
                )
                for key, value in parsed.items()
            }
            text = json.dumps(redacted, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    if len(text) > limit:
        return f"{text[:limit]}..."
    return text
