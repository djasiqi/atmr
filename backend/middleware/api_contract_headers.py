"""En-tetes de contrat d'API (X-API-Version) - sans Deprecation/Sunset (v1 = API supportee)."""

from __future__ import annotations

import os

from flask import request


def register_api_contract_headers(app) -> None:
    """Enregistre l'enrichissement des reponses HTTP pour les chemins /api/*."""

    @app.after_request
    def _add_api_version_headers(response):  # pyright: ignore
        p = request.path or ""
        if p.startswith("/api/"):
            response.headers["X-API-Version"] = "v1; contract=stable"
            _sha = (os.getenv("API_GIT_SHA") or "").strip()
            if _sha:
                _git_short = 7
                short = _sha[:_git_short] if len(_sha) >= _git_short else _sha
                response.headers["X-API-Build"] = (
                    short if str(short).startswith("git:") else f"git:{short}"
                )
        return response
