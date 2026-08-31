"""Snapshot acteur à la création d'une demande institution."""

from __future__ import annotations

from typing import Any

from flask import g

from models.user import User
from shared.user_display import resolve_actor_display_name

API_KEY_ACTOR_LABEL = "Clé API"


class ActorRequiredError(ValueError):
    """Aucun acteur serveur valide — ne pas INSERT."""


def apply_created_by_snapshot(
    transport_req: Any,
    *,
    user_id: int | None,
) -> None:
    """Pose ``created_by_user_id`` + ``created_by_display_name`` depuis le serveur.

    Les champs client sont ignorés : seuls ``user_id`` JWT ou l'auth API key comptent.
    """
    auth_method = g.get("auth_method")
    actor_user = User.query.get(user_id) if user_id is not None else None

    if user_id is not None and actor_user is None:
        raise ActorRequiredError("Acteur serveur introuvable")

    if auth_method == "api_key" and user_id is None:
        display = API_KEY_ACTOR_LABEL
    else:
        display = resolve_actor_display_name(actor_user, user_id=user_id)

    if not display or not str(display).strip():
        raise ActorRequiredError("Nom d'acteur serveur invalide")

    transport_req.created_by_user_id = user_id
    transport_req.created_by_display_name = str(display).strip()[:255]
