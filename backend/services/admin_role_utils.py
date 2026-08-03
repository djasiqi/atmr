"""Utilitaires de normalisation des rôles admin (PR1 Partenaires)."""

from __future__ import annotations

from typing import Any


def normalized_role_value(role: Any) -> str:
    """Retourne la valeur de rôle canonique en majuscules.

    Couvre ``UserRole.COMPANY``, alias ``UserRole.company``,
    ``\"COMPANY\"`` et ``\"company\"``.
    """
    value = role.value if hasattr(role, "value") else str(role or "")
    return value.strip().upper()
