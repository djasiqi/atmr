"""Normalisation des champs utilisateur partagés entre modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.user import User


def normalize_contact_email(value: str | None) -> str | None:
    """Normalise un email de contact (trim + lowercase). Chaîne vide → None."""
    if not value:
        return None
    value = value.strip().lower()
    return value or None


def find_user_by_normalized_email(
    normalized_email: str | None,
    *,
    exclude_user_id: int | None = None,
) -> User | None:
    """Recherche un utilisateur par email normalisé (détection de conflit).

    Ne filtre pas sur account_status : un email détenu par un compte
    archived ou disabled reste réservé (évite ambiguïté historique dans les audits).
    """
    if not normalized_email:
        return None

    from sqlalchemy import func

    from models.user import User

    query = User.query.filter(func.lower(User.email) == normalized_email)
    if exclude_user_id is not None:
        query = query.filter(User.id != exclude_user_id)
    return query.first()
