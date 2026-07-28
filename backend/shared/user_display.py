"""Affichage canonique du nom d'un utilisateur acteur (historique, audit)."""

from __future__ import annotations

import re
from typing import Any

_PLACEHOLDER_USER_RE = re.compile(r"^User #\d+$", re.IGNORECASE)


def is_placeholder_actor_display_name(name: str | None) -> bool:
    """True si le libellé est vide ou du type « User #91597 »."""
    if name is None:
        return True
    text = str(name).strip()
    if not text:
        return True
    return bool(_PLACEHOLDER_USER_RE.fullmatch(text))


def format_user_actor_display_name(
    *,
    user_id: int | None = None,
    user: Any | None = None,
    fallback: str | None = None,
    allow_db_lookup: bool = True,
) -> str | None:
    """Nom affichable pour un acteur d'historique.

    Ordre :
    1. prénom + nom
    2. username + ``(User #{id})`` si pas de nom complet
    3. email
    4. fallback fourni (ex. claim JWT) s'il n'est pas un placeholder
    5. ``User #{id}``
    """
    resolved_user = user
    resolved_id = user_id
    if resolved_user is not None and resolved_id is None:
        raw_id = getattr(resolved_user, "id", None)
        if raw_id is not None:
            try:
                resolved_id = int(raw_id)
            except (TypeError, ValueError):
                resolved_id = None

    if resolved_user is None and resolved_id is not None and allow_db_lookup:
        try:
            from models.user import User

            resolved_user = User.query.get(resolved_id)
        except Exception:
            resolved_user = None

    if resolved_user is not None:
        first = (getattr(resolved_user, "first_name", None) or "").strip()
        last = (getattr(resolved_user, "last_name", None) or "").strip()
        full = f"{first} {last}".strip()
        if full:
            return full

        username = (getattr(resolved_user, "username", None) or "").strip()
        if username:
            if resolved_id is not None:
                return f"{username} (User #{resolved_id})"
            return username

        email = (getattr(resolved_user, "email", None) or "").strip()
        if email:
            return email

    fallback_text = (str(fallback).strip() if fallback is not None else "") or ""
    if fallback_text and not is_placeholder_actor_display_name(fallback_text):
        return fallback_text

    if resolved_id is not None:
        return f"User #{resolved_id}"
    return None
