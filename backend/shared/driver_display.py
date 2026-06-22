"""Affichage canonique du nom chauffeur (évite « None None »)."""

from __future__ import annotations

from typing import Any


def format_driver_display_name(driver: Any | None) -> str | None:
    """Nom affichable d'un chauffeur : prénom/nom, username, ou repli #id."""
    if driver is None:
        return None
    try:
        user = getattr(driver, "user", None)
        if user is not None:
            first_name = (getattr(user, "first_name", None) or "").strip()
            last_name = (getattr(user, "last_name", None) or "").strip()
            full = f"{first_name} {last_name}".strip()
            if full:
                return full
            username = (getattr(user, "username", None) or "").strip()
            if username:
                return username
    except (AttributeError, TypeError):
        pass

    name = getattr(driver, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()

    driver_id = getattr(driver, "id", None)
    if driver_id is not None:
        return f"Chauffeur #{driver_id}"
    return None
