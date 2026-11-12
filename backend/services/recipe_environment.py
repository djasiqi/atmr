"""Identité de la base utilisée par une recette.

Le nom seul est exposé, jamais l'URL. La production ne publie pas cette identité.
Une écriture n'est refusée que si l'appel se déclare comme recette.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

RECIPE_DATABASE = "atmr_test"
RECIPE_REFUSED = "recette_base_non_autorisee"
RECIPE_REFUSED_MESSAGE = "Cette recette ne peut écrire que sur la base de test."


def _configured_database_url() -> str:
    try:
        from flask import current_app, has_app_context
    except ImportError:
        return str(os.environ.get("DATABASE_URL") or "").strip()
    if not has_app_context():
        return str(os.environ.get("DATABASE_URL") or "").strip()
    configured = current_app.config.get("SQLALCHEMY_DATABASE_URI")
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    return str(os.environ.get("DATABASE_URL") or "").strip()


def database_name(url: str | None = None) -> str | None:
    raw = (url if url is not None else _configured_database_url()).strip()
    if not raw:
        return None
    path = urlparse(raw).path.lstrip("/")
    name = path.split("/", 1)[0].strip()
    return name or None


def database_host(url: str | None = None) -> str | None:
    """Hôte de l'URL configurée. Jamais l'identifiant ni le secret."""
    raw = (url if url is not None else _configured_database_url()).strip()
    if not raw:
        return None
    return urlparse(raw).hostname


def recipe_writes_allowed(name: str | None = None) -> bool:
    return (name if name is not None else database_name()) == RECIPE_DATABASE


def production_environment(
    flask_config: str | None = None, environment: str | None = None
) -> bool:
    config_name = (
        flask_config
        if flask_config is not None
        else os.environ.get("FLASK_CONFIG") or os.environ.get("FLASK_ENV") or ""
    )
    runtime = (
        environment if environment is not None else os.environ.get("ENVIRONMENT") or ""
    )
    return str(config_name).strip().lower() in {"production", "prod"} or str(
        runtime
    ).strip().lower() in {
        "production",
        "prod",
    }


def recipe_target_payload(name: str | None = None) -> dict[str, object]:
    resolved = database_name() if name is None else name
    return {
        "database": resolved,
        "writable_recipe": resolved == RECIPE_DATABASE,
    }


def disclose_driver_settlement_panel(
    payload: dict[str, object], *, panel_enabled: bool
) -> dict[str, object]:
    """Le client ne débloque pas le panneau : seul le serveur le déclare."""
    disclosed = dict(payload)
    disclosed["driver_settlement_panel"] = bool(panel_enabled)
    return disclosed


def recipe_trial_must_block(
    method: str,
    header: str | None,
    name: str | None,
    *,
    local_recipe: bool = False,
) -> bool:
    """Bloque une écriture déclarée recette hors base de test et hors recette locale.

    `local_recipe` est décidé par le serveur (base jetable, hors production,
    interrupteur SETTLEMENT_LOCAL_RECIPE). Un en-tête client ne l'ouvre pas.
    """
    if str(method or "").upper() in {"GET", "HEAD", "OPTIONS"}:
        return False
    if str(header or "").strip() != "1":
        return False
    if name == RECIPE_DATABASE:
        return False
    return not local_recipe
