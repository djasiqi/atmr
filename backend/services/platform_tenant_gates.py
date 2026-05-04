"""Gates runtime : suspension tenant plateforme (back-end autoritative)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from ext import db
from models.company import Company
from models.enums import UserRole
from models.user import User
from services.platform_exceptions import PlatformTenantSuspended
from services.platform_governance_constants import ERROR_TENANT_PLATFORM_SUSPENDED


def _company_for_user(user: User) -> Company | None:
    if user.role == UserRole.company:
        c = getattr(user, "company", None)
        if c is not None:
            return c
        return db.session.scalar(select(Company).where(Company.user_id == user.id))
    if user.role == UserRole.driver and getattr(user, "driver", None):
        cid = getattr(user.driver, "company_id", None)
        if cid:
            return db.session.get(Company, int(cid))
    return None


def gate_login_if_company_suspended(user: User) -> dict[str, Any] | None:
    """
    Après authentification réussie : bloquer émission session si tenant suspendu.
    Retourne un dict d'erreur JSON + code HTTP 403 si bloqué, sinon None.
    """
    company = _company_for_user(user)
    if company is None:
        return None
    if not company.platform_suspended:
        return None
    return {
        "error": "Ce transporteur est suspendu au sens plateforme.",
        "error_code": ERROR_TENANT_PLATFORM_SUSPENDED,
        "reason_code": ERROR_TENANT_PLATFORM_SUSPENDED,
        "human_reason": "Ce transporteur est suspendu au sens plateforme.",
        "retryable": False,
    }


def assert_company_not_platform_suspended(company_id: int) -> None:
    """Raise PlatformTenantSuspended si company.platform_suspended."""
    c = db.session.get(Company, int(company_id))
    if c is None:
        return
    if c.platform_suspended:
        raise PlatformTenantSuspended()


def gate_gps_ingestion_allowed(company_id: int) -> bool:
    """
    GPS sous suspension : V1 — ingestion autorisée pour observabilité (décision plateforme).
    Les mutations métier restent bloquées via assert_company_not_platform_suspended.
    """
    _ = company_id
    return True
