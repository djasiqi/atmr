"""Effective access diagnostique (shadow only) — CP-PR1."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from ext import db
from models.control_plane import (
    OrganizationMembership,
    OrganizationServiceEntitlement,
    PermissionCatalog,
    PlatformOrganization,
    RoleTemplate,
    RoleTemplatePermission,
    ServiceCatalog,
)
from models.enums import UserRole
from models.user import User

_ROLE_EXPECTED: dict[str, str] = {
    "institution_admin": "INSTITUTION",
    "institution_requester": "INSTITUTION",
    "institution_reader": "INSTITUTION",
    "institution_billing": "INSTITUTION",
    "institution_curator": "INSTITUTION",
    "institution_reception": "INSTITUTION",
    "legacy_unresolved": "INSTITUTION",
    "company_owner": "COMPANY",
    "company_driver": "DRIVER",
    "company_dispatcher": "COMPANY",
    "company_billing": "COMPANY",
    "company_reader": "COMPANY",
}

_STATE_RANK = {"blocked": 3, "needs_review": 2, "eligible": 1}


def _raise_state(current: str, candidate: str) -> str:
    if _STATE_RANK.get(candidate, 0) > _STATE_RANK.get(current, 0):
        return candidate
    return current


def _role_value(user: User) -> str:
    role = getattr(user, "role", None)
    if role is None:
        return ""
    if isinstance(role, UserRole):
        return role.value
    return str(role).upper()


def compute_effective_access(user_id: int) -> dict[str, Any]:
    user = db.session.get(User, user_id)
    if user is None:
        return {
            "decision_mode": "shadow",
            "subject_state": "blocked",
            "permissions_detected": [],
            "permissions_enforced": [],
            "blocking_reasons": [{"code": "ACCOUNT_NOT_FOUND"}],
            "memberships": [],
            "scope": {"type": "none", "schema_version": 1, "value": {}},
        }

    blocking: list[dict[str, Any]] = []
    subject_state = "eligible"

    if getattr(user, "archived_at", None) is not None:
        blocking.append({"code": "ACCOUNT_ARCHIVED"})
        subject_state = "blocked"
    if (
        getattr(user, "disabled_at", None) is not None
        or user.account_status == "disabled"
    ):
        blocking.append({"code": "ACCOUNT_DISABLED"})
        subject_state = "blocked"
    if user.account_status in ("invited", "pending_activation"):
        blocking.append({"code": "ACCOUNT_NOT_ACTIVATED"})
        subject_state = "blocked"
    if bool(getattr(user, "force_password_change", False)):
        blocking.append({"code": "PASSWORD_CHANGE_REQUIRED"})
        subject_state = _raise_state(subject_state, "blocked")

    memberships = db.session.scalars(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.membership_status != "removed",
        )
    ).all()

    membership_payloads: list[dict[str, Any]] = []
    all_detected: list[str] = []
    has_active_membership = False

    if not memberships and subject_state != "blocked":
        blocking.append({"code": "NO_ACTIVE_MEMBERSHIP"})
        subject_state = "blocked"

    for m in memberships:
        org = db.session.get(PlatformOrganization, m.organization_id)
        if org is None:
            continue
        m_block: list[dict[str, Any]] = []

        if m.membership_status == "invited":
            m_block.append({"code": "MEMBERSHIP_INVITED"})
            subject_state = _raise_state(subject_state, "blocked")
        elif m.membership_status == "needs_review":
            m_block.append({"code": "MEMBERSHIP_NEEDS_REVIEW"})
            subject_state = _raise_state(subject_state, "needs_review")
        elif m.membership_status != "active":
            m_block.append({"code": "MEMBERSHIP_NOT_ACTIVE"})
            subject_state = _raise_state(subject_state, "blocked")
        else:
            has_active_membership = True

        if org.lifecycle_status != "active":
            m_block.append(
                {
                    "code": "ORGANIZATION_NOT_ACTIVE",
                    "lifecycle_status": org.lifecycle_status,
                }
            )
            subject_state = _raise_state(subject_state, "blocked")

        role = (
            db.session.get(RoleTemplate, m.role_template_id)
            if m.role_template_id
            else None
        )
        role_key = role.role_key if role else None
        if role is None:
            m_block.append({"code": "ROLE_TEMPLATE_MISSING"})
            subject_state = _raise_state(subject_state, "needs_review")

        expected = _ROLE_EXPECTED.get(role_key or "", "")
        actual = _role_value(user)
        if expected and actual != expected:
            m_block.append(
                {
                    "code": "LEGACY_ROLE_MISMATCH",
                    "expected": expected,
                    "actual": actual,
                }
            )
            subject_state = _raise_state(subject_state, "blocked")

        detected: list[str] = []
        can_detect = (
            role is not None
            and m.membership_status == "active"
            and org.lifecycle_status == "active"
            and not any(b["code"] == "LEGACY_ROLE_MISMATCH" for b in m_block)
            and not any(
                b["code"]
                in (
                    "ACCOUNT_ARCHIVED",
                    "ACCOUNT_DISABLED",
                    "ACCOUNT_NOT_ACTIVATED",
                    "PASSWORD_CHANGE_REQUIRED",
                )
                for b in blocking
            )
        )
        if can_detect:
            now = datetime.now(UTC)
            entitlements = db.session.scalars(
                select(OrganizationServiceEntitlement).where(
                    OrganizationServiceEntitlement.organization_id == org.id,
                    OrganizationServiceEntitlement.status.in_(("trial", "enabled")),
                )
            ).all()
            enabled_keys: set[str] = set()
            for e in entitlements:
                if e.starts_at and e.starts_at > now:
                    continue
                if e.ends_at and e.ends_at < now:
                    continue
                svc = db.session.get(ServiceCatalog, e.service_catalog_id)
                if svc:
                    enabled_keys.add(svc.service_key)

            perm_ids = db.session.scalars(
                select(RoleTemplatePermission.permission_catalog_id).where(
                    RoleTemplatePermission.role_template_id == role.id
                )
            ).all()
            for pid in perm_ids:
                perm = db.session.get(PermissionCatalog, pid)
                if perm is None:
                    continue
                if (
                    perm.required_service_key is None
                    or perm.required_service_key in enabled_keys
                ):
                    detected.append(perm.permission_key)

        all_detected.extend(detected)
        membership_payloads.append(
            {
                "organization_public_id": str(org.public_id),
                "organization_type": org.organization_type,
                "role": role_key,
                "membership_status": m.membership_status,
                "permissions_detected": detected,
                "blocking_reasons": m_block,
                "scope": {
                    "type": m.scope_type,
                    "schema_version": m.scope_schema_version,
                    "value": m.scope_json or {},
                },
            }
        )

    if memberships and not has_active_membership and subject_state == "eligible":
        subject_state = "blocked"
        blocking.append({"code": "NO_ACTIVE_MEMBERSHIP"})

    return {
        "decision_mode": "shadow",
        "subject_state": subject_state,
        "account_status": user.account_status,
        "data_origin": getattr(user, "data_origin", "unknown"),
        "permissions_detected": sorted(set(all_detected)),
        "permissions_enforced": [],
        "blocking_reasons": blocking,
        "memberships": membership_payloads,
        "scope": (
            membership_payloads[0]["scope"]
            if len(membership_payloads) == 1
            else {"type": "multi", "schema_version": 1, "value": {}}
        ),
    }
