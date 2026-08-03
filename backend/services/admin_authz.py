"""Autorisations granulaires admin.* (PR2bis + durcissement).

Réutilise les grants `platform_admin_permission_grant` (permissions admin.*).

Mode compatibilité (ADMIN_CAPABILITIES_ENFORCED=false, défaut) :
  - accès effectif = toutes les capacités (rôle admin legacy) ;
  - politique simulée = grants présents (logs « aurait refusé » si grant partiel).

Mode enforced (ADMIN_CAPABILITIES_ENFORCED=true) :
  - accès effectif = grants uniquement ;
  - sans grants admin.* ⇒ aucune capacité (matrice explicite obligatoire).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from flask import jsonify
from sqlalchemy import select

from ext import db
from models.enums import UserRole
from models.platform_admin_permission_grant import PlatformAdminPermissionGrant
from models.user import User

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

CAP_OVERVIEW_READ = "admin.overview.read"
CAP_BOOKINGS_READ = "admin.bookings.read"
CAP_BOOKINGS_EXPORT = "admin.bookings.export"
CAP_PARTNERS_READ = "admin.partners.read"
CAP_USERS_MANAGE = "admin.users.manage"
CAP_USERS_SECURITY = "admin.users.security"
CAP_BILLING_READ = "admin.billing.read"
CAP_BILLING_LOCK = "admin.billing.lock"
CAP_BILLING_ISSUE = "admin.billing.issue"
CAP_BILLING_VALIDATE = "admin.billing.validate"
CAP_CONFIGURATION_MANAGE = "admin.configuration.manage"
CAP_LABS_READ = "admin.labs.read"
CAP_LABS_EXECUTE = "admin.labs.execute"

ALL_ADMIN_CAPABILITIES: frozenset[str] = frozenset(
    {
        CAP_OVERVIEW_READ,
        CAP_BOOKINGS_READ,
        CAP_BOOKINGS_EXPORT,
        CAP_PARTNERS_READ,
        CAP_USERS_MANAGE,
        CAP_USERS_SECURITY,
        CAP_BILLING_READ,
        CAP_BILLING_LOCK,
        CAP_BILLING_ISSUE,
        CAP_BILLING_VALIDATE,
        CAP_CONFIGURATION_MANAGE,
        CAP_LABS_READ,
        CAP_LABS_EXECUTE,
    }
)

ADMIN_IMPLIED_CAPABILITIES: frozenset[str] = ALL_ADMIN_CAPABILITIES


def admin_capabilities_enforced() -> bool:
    """True uniquement si ADMIN_CAPABILITIES_ENFORCED est explicitement activé."""
    raw = (os.getenv("ADMIN_CAPABILITIES_ENFORCED") or "false").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _admin_capability_grants(user_id: int) -> frozenset[str]:
    rows = db.session.scalars(
        select(PlatformAdminPermissionGrant.permission).where(
            PlatformAdminPermissionGrant.user_id == user_id,
            PlatformAdminPermissionGrant.permission.like("admin.%"),
        )
    ).all()
    return frozenset(rows)


def user_policy_admin_capabilities(user_id: int) -> frozenset[str]:
    """Politique simulée = grants admin.* uniquement (peut être vide)."""
    u = db.session.get(User, user_id)
    if not u or u.role != UserRole.ADMIN:
        return frozenset()
    return _admin_capability_grants(user_id)


def user_effective_admin_capabilities(user_id: int) -> frozenset[str]:
    """Capacités effectivement accordées pour l'UI / le contrôle d'accès.

    Compat : ensemble complet.
    Enforced : grants uniquement (vide si aucune matrice explicite).
    """
    u = db.session.get(User, user_id)
    if not u or u.role != UserRole.ADMIN:
        return frozenset()
    if admin_capabilities_enforced():
        return _admin_capability_grants(user_id)
    return ADMIN_IMPLIED_CAPABILITIES


def user_has_admin_capability(user_id: int | None, capability: str) -> bool:
    """Décide l'accès (tient compte de ADMIN_CAPABILITIES_ENFORCED)."""
    if user_id is None:
        return False
    u = db.session.get(User, user_id)
    if not u or u.role != UserRole.ADMIN:
        return False

    if not admin_capabilities_enforced():
        policy = user_policy_admin_capabilities(user_id)
        if policy and capability not in policy:
            logger.info(
                "admin_capability_would_deny user_id=%s capability=%s enforced=false "
                "decision=allow_legacy",
                user_id,
                capability,
            )
        return True

    grants = _admin_capability_grants(user_id)
    if capability in grants:
        return True
    logger.info(
        "admin_capability_denied user_id=%s capability=%s enforced=true",
        user_id,
        capability,
    )
    return False


def require_admin_capability(capability: str) -> Callable[[F], F]:
    """Décorateur Flask : exige une capacité admin.* (après jwt + rôle admin)."""

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            from shared.infrastructure.adapters.auth_adapter import (
                get_current_user_via_use_case,
            )

            user = get_current_user_via_use_case()
            if not user:
                return jsonify(
                    {"error": "unauthorized", "message": "Utilisateur introuvable."}
                ), 401
            if not user_has_admin_capability(user.id, capability):
                return jsonify(
                    {
                        "error": "forbidden",
                        "message": "Capacité administrateur insuffisante.",
                        "capability": capability,
                    }
                ), 403
            return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def capabilities_payload_for_user(user_id: int) -> dict[str, Any]:
    """Payload API pour le frontend (hook useAdminCapabilities)."""
    enforced = admin_capabilities_enforced()
    effective = sorted(user_effective_admin_capabilities(user_id))
    policy = sorted(user_policy_admin_capabilities(user_id))
    return {
        "enforced": enforced,
        "capabilities_effective": effective,
        "capabilities_policy": policy,
        "note": (
            "ADMIN_CAPABILITIES_ENFORCED=false : compat rôle admin (accès complet) ; "
            "capabilities_policy reflète les grants pour simulation."
            if not enforced
            else "ADMIN_CAPABILITIES_ENFORCED=true : accès = grants admin.* uniquement "
            "(matrice explicite obligatoire)."
        ),
    }
