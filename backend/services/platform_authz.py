"""Constantes capability bundles / permissions plateforme (V1).

Grants persistés : `platform_admin_permission_grant` (voir migration 20260329_plat_admin_perm).
Aligné sur docs/platform/spec-normative-v1.md et DECISIONS.md.
"""

from __future__ import annotations

from sqlalchemy import select

from ext import db
from models.enums import UserRole
from models.platform_admin_permission_grant import PlatformAdminPermissionGrant
from models.user import User

# Permissions nommées (granularité logique)
PERM_OBSERVE_TENANT_READ = "observe.tenant.read"
PERM_GOVERNANCE_TENANT_SUSPEND = "governance.tenant.suspend"
PERM_POLICY_EXPLAIN = "policy.explain"
PERM_OPERATE_RUNBOOKS = "operate.runbooks.execute"

# Bundles référencés par la spec (assignation DB — LATER)
BUNDLE_OBSERVE_CORE = "observe_core"
BUNDLE_OPERATE_TENANT_CONTROLS = "operate_tenant_controls"
BUNDLE_APPROVE_PROD_CHANGES = "approve_prod_changes"

BUNDLE_PERMISSIONS: dict[str, frozenset[str]] = {
    BUNDLE_OBSERVE_CORE: frozenset({PERM_OBSERVE_TENANT_READ, PERM_POLICY_EXPLAIN}),
    BUNDLE_OPERATE_TENANT_CONTROLS: frozenset(
        {
            PERM_OBSERVE_TENANT_READ,
            PERM_GOVERNANCE_TENANT_SUSPEND,
            PERM_POLICY_EXPLAIN,
        }
    ),
    BUNDLE_APPROVE_PROD_CHANGES: frozenset(
        {PERM_GOVERNANCE_TENANT_SUSPEND, PERM_OPERATE_RUNBOOKS}
    ),
}

# Rôle applicatif actuel : un seul rôle admin couvre les bundles pilote (slice V1).
ADMIN_IMPLIED_PERMISSIONS: frozenset[str] = frozenset(
    {
        PERM_OBSERVE_TENANT_READ,
        PERM_GOVERNANCE_TENANT_SUSPEND,
        PERM_POLICY_EXPLAIN,
        PERM_OPERATE_RUNBOOKS,
    }
)


def admin_has_permission(permission: str) -> bool:
    """Compat tests / code legacy : ensemble pilote complet sans résolution par utilisateur."""
    return permission in ADMIN_IMPLIED_PERMISSIONS


def user_effective_platform_permissions(user_id: int) -> frozenset[str]:
    """Permissions effectives pour un utilisateur admin (grants DB ou fallback legacy)."""
    u = db.session.get(User, user_id)
    if not u or u.role != UserRole.ADMIN:
        return frozenset()
    rows = db.session.scalars(
        select(PlatformAdminPermissionGrant.permission).where(
            PlatformAdminPermissionGrant.user_id == user_id
        )
    ).all()
    if rows:
        return frozenset(rows)
    return ADMIN_IMPLIED_PERMISSIONS


def user_has_platform_permission(user_id: int | None, permission: str) -> bool:
    """Vérifie une permission nommée pour un admin (grants persistés)."""
    if user_id is None:
        return False
    return permission in user_effective_platform_permissions(user_id)
