"""Kill-switch du plan de contrôle ML (F-04 / F-05).

Contrôle l'accès aux préfixes :
- /api/feature-flags
- /api/shadow-mode
- /api/ml-monitoring
"""

from __future__ import annotations

import logging
import os
from typing import Any

from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)

ML_CONTROL_PLANE_PREFIXES = (
    "/api/feature-flags",
    "/api/shadow-mode",
    "/api/ml-monitoring",
)

ML_CONTROL_PLANE_DISABLED_ERROR = "ml_control_plane_disabled"
ML_CONTROL_PLANE_DISABLED_MESSAGE = "ML control plane API is disabled"


def is_ml_control_plane_api_enabled() -> bool:
    """Retourne True si le plan de contrôle ML est activé.

    - Variable absente → activé (dev/tests)
    - true/1/yes → activé
    - false/0/no → désactivé
    - toute autre valeur → désactivé (fail-closed)
    """
    raw = os.getenv("ML_CONTROL_PLANE_API_ENABLED")
    if raw is None:
        return True
    value = raw.strip().lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    logger.warning(
        "[MLControlPlane] Valeur invalide ML_CONTROL_PLANE_API_ENABLED=%r → fail-closed",
        raw,
    )
    return False


def ml_control_plane_disabled_payload() -> dict[str, str]:
    return {
        "error": ML_CONTROL_PLANE_DISABLED_ERROR,
        "message": ML_CONTROL_PLANE_DISABLED_MESSAGE,
    }


def path_matches_ml_control_plane(path: str) -> bool:
    path = path or ""
    return any(
        path == prefix or path.startswith(prefix + "/")
        for prefix in ML_CONTROL_PLANE_PREFIXES
    )


def register_ml_control_plane_kill_switch(app: Flask) -> None:
    """Enregistre le before_request kill-switch (à appeler avant CSRF)."""

    @app.before_request
    def _ml_control_plane_kill_switch() -> Any:  # pyright: ignore[reportUnusedFunction]
        if not path_matches_ml_control_plane(request.path or ""):
            return None
        if is_ml_control_plane_api_enabled():
            return None
        return jsonify(ml_control_plane_disabled_payload()), 503
