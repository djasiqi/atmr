"""Capacités privilégiées — enrollment MFA obligatoire.

Ne pas assimiler ``UserRole.COMPANY`` / ``UserRole.INSTITUTION`` à un admin.
Un demandeur ou un lecteur institution ne doit pas être forcé.
"""

from __future__ import annotations

from typing import Any

from models.enums import InstitutionRole, UserRole


def _role_value(role: Any) -> str | None:
    if role is None:
        return None
    return role.value if hasattr(role, "value") else str(role)


def _is_platform_admin(user: Any) -> bool:
    return _role_value(getattr(user, "role", None)) == UserRole.ADMIN.value


def _is_institution_admin(user: Any) -> bool:
    raw = getattr(user, "institution_role", None)
    if raw is None:
        return False
    return _role_value(raw) == InstitutionRole.ADMIN.value


def _is_company_tenant_owner(user: Any) -> bool:
    """True si l'utilisateur est l'owner réel d'au moins une entreprise."""
    user_id = getattr(user, "id", None)
    if user_id is None:
        return False
    company = getattr(user, "company", None)
    if company is not None:
        return getattr(company, "user_id", None) == user_id
    try:
        from models.company import Company

        return Company.query.filter_by(user_id=user_id).first() is not None
    except Exception:
        return False


def requires_mfa_enrollment(user: Any) -> bool:
    """Enrollment TOTP obligatoire pour les comptes réellement privilégiés.

    - toujours : super-admin LIRIE (``UserRole.ADMIN``)
    - institution : ``InstitutionRole.ADMIN`` uniquement
    - entreprise : owner tenant (``Company.user_id``), pas tout ``UserRole.COMPANY``
    - jamais : DRIVER, CLIENT, REQUESTER, READER, RECEPTION, CURATOR, BILLING
    """
    if user is None:
        return False
    if _is_platform_admin(user):
        return True
    if _is_institution_admin(user):
        return True
    return _is_company_tenant_owner(user)


def is_privileged_account(user: Any) -> bool:
    """Alias explicite pour les règles de session / disable TOTP."""
    return requires_mfa_enrollment(user)
