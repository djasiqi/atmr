"""Garde d'accès tenant company + audit ADMIN cross-tenant (Lot 0 P0)."""

from __future__ import annotations

import logging
from typing import Any

from flask import request
from flask_jwt_extended import get_jwt_identity

from models import User
from models.enums import UserRole
from security.audit_log import AuditLogger
from security.authorization import AuthorizationService

logger = logging.getLogger(__name__)


def get_current_user_from_jwt() -> User | None:
    """Charge l'utilisateur authentifié depuis le claim JWT (public_id)."""
    identity = get_jwt_identity()
    if not identity:
        return None
    return User.query.filter_by(public_id=identity).first()


def assert_company_access(
    company_id: int,
    *,
    resource: str,
    user: User | None = None,
) -> tuple[User | None, tuple[dict[str, Any], int] | None]:
    """Vérifie l'accès company via AuthorizationService.

    ADMIN cross-tenant autorisé avec journalisation audit.
    COMPANY limité à sa propre entreprise.

    Returns:
        (user, None) si accès OK
        (user|None, (error_body, status_code)) si refus
    """
    if user is None:
        user = get_current_user_from_jwt()
    if not user:
        return None, ({"error": "Utilisateur non trouvé"}, 401)

    has_access, err = AuthorizationService.check_company_resource_access(
        company_id, user
    )
    if not has_access:
        return user, err

    # Journaliser chaque accès ADMIN cross-tenant
    if user.role == UserRole.ADMIN:
        own_company = None
        try:
            from models import Company

            own_company = Company.query.filter_by(user_id=user.id).first()
        except Exception:
            logger.debug("Impossible de résoudre company ADMIN pour audit", exc_info=True)

        is_cross_tenant = own_company is None or own_company.id != company_id
        if is_cross_tenant:
            try:
                AuditLogger.log_action(
                    action_type="admin_cross_tenant_access",
                    action_category="security",
                    user_id=user.id,
                    user_type="admin",
                    result_status="success",
                    action_details={
                        "resource": resource,
                        "target_company_id": company_id,
                    },
                    company_id=company_id,
                    ip_address=request.remote_addr if request else None,
                    user_agent=request.headers.get("User-Agent") if request else None,
                )
            except Exception as audit_err:
                logger.warning(
                    "Échec audit admin_cross_tenant_access resource=%s: %s",
                    resource,
                    audit_err,
                )

    return user, None
