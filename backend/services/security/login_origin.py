"""Validation Origin/Referer pour le login web (Lot 1 — L1-D)."""

from __future__ import annotations

import os
from urllib.parse import urlparse

from flask import request


def _is_production() -> bool:
    env = (
        (
            os.getenv("ENVIRONMENT")
            or os.getenv("FLASK_CONFIG")
            or os.getenv("FLASK_ENV")
            or ""
        )
        .strip()
        .lower()
    )
    return env == "production"


def _allowed_origins() -> set[str]:
    raw = (
        os.getenv("LOGIN_ALLOWED_ORIGINS")
        or os.getenv("SOCKETIO_CORS_ORIGINS")
        or os.getenv("CORS_ORIGINS")
        or ""
    ).strip()
    origins: set[str] = set()
    if raw and raw != "*":
        for part in raw.split(","):
            o = part.strip().rstrip("/")
            if o:
                origins.add(o)
    frontend = (
        (os.getenv("FRONTEND_URL") or os.getenv("PUBLIC_FRONTEND_URL") or "")
        .strip()
        .rstrip("/")
    )
    if frontend:
        origins.add(frontend)
    return origins


def _origin_from_referer(referer: str) -> str | None:
    try:
        parsed = urlparse(referer)
        if not parsed.scheme or not parsed.netloc:
            return None
        return f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
    except Exception:
        return None


def extract_request_origin() -> str | None:
    """Origin header, sinon repli Referer (même forme scheme://host[:port])."""
    origin = (request.headers.get("Origin") or "").strip().rstrip("/")
    if origin:
        return origin
    referer = (request.headers.get("Referer") or "").strip()
    if referer:
        return _origin_from_referer(referer)
    return None


def validate_login_origin_for_web() -> tuple[bool, str | None]:
    """Valide Origin/Referer pour login web.

    Returns:
        (ok, error_code) — error_code parmi missing_origin / origin_not_allowed
    """
    try:
        from flask import current_app

        if current_app and bool(current_app.config.get("TESTING", False)):
            return True, None
    except Exception:
        pass

    # Contrôle web uniquement. Le bypass mobile (Bearer) se fait à l'appelant
    # (routes.auth.Login) via _is_mobile_request — pas ici.
    origin = extract_request_origin()
    allowed = _allowed_origins()

    if not origin:
        if _is_production():
            return False, "missing_origin"
        # Hors prod : tolérant si whitelist vide
        return True, None

    if not allowed:
        if _is_production():
            return False, "origin_not_allowed"
        return True, None

    if origin.rstrip("/") not in {a.rstrip("/") for a in allowed}:
        return False, "origin_not_allowed"
    return True, None
