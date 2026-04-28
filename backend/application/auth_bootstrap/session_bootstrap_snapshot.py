"""Snapshot interne plat pour la lecture bootstrap (pas de relations ORM)."""

from __future__ import annotations

from dataclasses import dataclass

from models.enums import UserRole


@dataclass(frozen=True, slots=True)
class SessionBootstrapSnapshot:
    """Données minimales pour appliquer les mêmes règles que `_check_user_profile_active`."""

    user_id: int
    public_id: str
    username: str
    email: str | None
    role: UserRole
    account_status: str | None
    driver_id: int | None
    driver_company_id: int | None
    driver_is_active: bool | None
    company_relation_id: int | None
    client_active_flags: tuple[bool, ...]
