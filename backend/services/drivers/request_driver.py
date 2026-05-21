"""Résolution du chauffeur actif pour une requête HTTP (app unifiée multi-contexte)."""

from __future__ import annotations

from typing import Any

from flask import request

from models import Driver


def _parse_context_driver_id(active_ctx: str) -> int | None:
    if not active_ctx.startswith("driver:"):
        return None
    try:
        return int(active_ctx.split(":", 1)[1].strip())
    except (ValueError, IndexError):
        return None


def _user_role_upper(user) -> str:
    return str(getattr(user.role, "value", user.role)).upper()


def resolve_request_driver(
    user,
    *,
    active_context_id: str | None = None,
) -> tuple[Driver | None, tuple[dict[str, Any], int] | None]:
    """Retourne le chauffeur actif pour la requête courante.

    - Rôle BDD ``DRIVER`` → fiche chauffeur liée au user.
    - App unifiée (rôle BDD ``COMPANY``, etc.) → fiche chauffeur si
      ``X-Active-Context-Id`` vaut ``driver:{id}`` correspondant.
    """
    if user is None:
        return None, ({"error": "Utilisateur introuvable"}, 404)

    driver = getattr(user, "driver", None)
    if driver is None:
        from repositories.driver_repository import DriverRepository

        driver = DriverRepository().find_model_by_user_id(int(user.id))

    if active_context_id is None:
        active_ctx = (request.headers.get("X-Active-Context-Id") or "").strip()
    else:
        active_ctx = active_context_id.strip()
    context_driver_id = _parse_context_driver_id(active_ctx)

    if _user_role_upper(user) == "DRIVER":
        if not driver:
            return None, ({"error": "Profil chauffeur introuvable"}, 404)
        return driver, None

    if (
        context_driver_id is not None
        and driver is not None
        and int(driver.id) == context_driver_id
    ):
        return driver, None

    return None, ({"error": "Réservé aux chauffeurs"}, 403)
