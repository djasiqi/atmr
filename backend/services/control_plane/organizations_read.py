"""Lecture organisations control plane + modes legacy/compare/control_plane."""

from __future__ import annotations

import os
from typing import Any
from uuid import UUID

from sqlalchemy import func, select

from ext import db
from models.company import Company
from models.control_plane import (
    ControlPlaneAnomaly,
    OrganizationMembership,
    OrganizationServiceEntitlement,
    PlatformOrganization,
    ServiceCatalog,
)
from models.institution import Institution
from models.refresh_token import RefreshToken
from models.user import User
from services.admin_partners_organizations import (
    list_partner_organizations,
)


def organizations_read_mode() -> str:
    raw = (os.getenv("CONTROL_PLANE_ORGANIZATIONS_READ_MODE") or "legacy").strip().lower()
    if raw in ("legacy", "compare", "control_plane"):
        return raw
    return "legacy"


def _contains_non_production(org_id: int) -> bool:
    exists = db.session.scalar(
        select(OrganizationMembership.id)
        .join(User, User.id == OrganizationMembership.user_id)
        .where(
            OrganizationMembership.organization_id == org_id,
            User.data_origin != "production",
        )
        .limit(1)
    )
    return exists is not None


def _services_detected(org_id: int) -> list[dict[str, Any]]:
    rows = db.session.execute(
        select(ServiceCatalog.service_key, ServiceCatalog.label, OrganizationServiceEntitlement.enforcement_mode)
        .join(
            OrganizationServiceEntitlement,
            OrganizationServiceEntitlement.service_catalog_id == ServiceCatalog.id,
        )
        .where(
            OrganizationServiceEntitlement.organization_id == org_id,
            OrganizationServiceEntitlement.status.in_(("trial", "enabled")),
        )
    ).all()
    return [
        {
            "service_key": r[0],
            "label": r[1],
            "enforcement_mode": r[2],
        }
        for r in rows
    ]


def _users_count(org_id: int) -> int:
    return (
        db.session.scalar(
            select(func.count())
            .select_from(OrganizationMembership)
            .where(
                OrganizationMembership.organization_id == org_id,
                OrganizationMembership.membership_status != "removed",
            )
        )
        or 0
    )


def _last_activity_for_org(org_id: int) -> str | None:
    user_ids = db.session.scalars(
        select(OrganizationMembership.user_id).where(
            OrganizationMembership.organization_id == org_id,
            OrganizationMembership.membership_status != "removed",
        )
    ).all()
    if not user_ids:
        return None
    last = db.session.scalar(
        select(func.max(RefreshToken.last_used_at)).where(
            RefreshToken.user_id.in_(list(user_ids))
        )
    )
    return last.isoformat() if last else None


def _serialize_cp_org(org: PlatformOrganization) -> dict[str, Any]:
    name = None
    contact_email = None
    if org.organization_type == "company" and org.company_id:
        company = db.session.get(Company, org.company_id)
        if company:
            name = company.name
            contact_email = company.contact_email
    elif org.organization_type == "institution" and org.institution_id:
        inst = db.session.get(Institution, org.institution_id)
        if inst:
            name = inst.name
            contact_email = inst.contact_email

    services = _services_detected(int(org.id))
    return {
        "public_id": str(org.public_id),
        "organization_type": org.organization_type,
        "organization_id": org.company_id or org.institution_id,
        "name": name,
        "contact_email": contact_email,
        "lifecycle_status": org.lifecycle_status,
        "lifecycle_source": org.lifecycle_source,
        "data_origin": org.data_origin,
        "data_origin_confidence": org.data_origin_confidence,
        "contains_non_production_accounts": _contains_non_production(int(org.id)),
        "accounts_count": _users_count(int(org.id)),
        "services_detected_count": len(services),
        "services_detected": services,
        "services_label": "Prestations détectées",
        "last_activity_at": _last_activity_for_org(int(org.id)),
        "read_source": "control_plane",
        "created_at": org.created_at.isoformat() if org.created_at else None,
    }


def list_control_plane_organizations(
    *,
    page: int = 1,
    per_page: int = 25,
    include_synthetic: bool = False,
    organization_type: str | None = None,
    lifecycle_status: str | None = None,
    data_origin: str | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    page = max(1, page)
    per_page = min(max(1, per_page), 100)
    query = select(PlatformOrganization)
    if not include_synthetic:
        query = query.where(PlatformOrganization.data_origin == "production")
    if organization_type in ("company", "institution"):
        query = query.where(PlatformOrganization.organization_type == organization_type)
    if lifecycle_status:
        query = query.where(PlatformOrganization.lifecycle_status == lifecycle_status)
    if data_origin:
        query = query.where(PlatformOrganization.data_origin == data_origin)

    total = db.session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.session.scalars(
        query.order_by(PlatformOrganization.id.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
    ).all()

    items = [_serialize_cp_org(o) for o in rows]
    if search:
        ql = search.strip().lower()
        items = [
            i
            for i in items
            if (i.get("name") or "").lower().find(ql) >= 0
            or (i.get("contact_email") or "").lower().find(ql) >= 0
        ]

    summary = {
        "organizations_production": db.session.scalar(
            select(func.count())
            .select_from(PlatformOrganization)
            .where(PlatformOrganization.data_origin == "production")
        )
        or 0,
        "active": db.session.scalar(
            select(func.count())
            .select_from(PlatformOrganization)
            .where(
                PlatformOrganization.data_origin == "production",
                PlatformOrganization.lifecycle_status == "active",
            )
        )
        or 0,
        "onboarding": db.session.scalar(
            select(func.count())
            .select_from(PlatformOrganization)
            .where(
                PlatformOrganization.data_origin == "production",
                PlatformOrganization.lifecycle_status == "onboarding",
            )
        )
        or 0,
        "suspended": db.session.scalar(
            select(func.count())
            .select_from(PlatformOrganization)
            .where(
                PlatformOrganization.data_origin == "production",
                PlatformOrganization.lifecycle_status == "suspended",
            )
        )
        or 0,
        "needs_attention": db.session.scalar(
            select(func.count())
            .select_from(ControlPlaneAnomaly)
            .where(
                ControlPlaneAnomaly.resolved_at.is_(None),
                ControlPlaneAnomaly.severity.in_(("critical", "warning")),
            )
        )
        or 0,
    }

    return {
        "items": items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": int(total),
            "pages": max(1, (int(total) + per_page - 1) // per_page),
        },
        "summary": summary,
        "read_mode": "control_plane",
    }


def _cp_public_id_for_legacy(org_type: str, org_id: int | None) -> str | None:
    if org_id is None:
        return None
    if org_type == "company":
        org = db.session.scalar(
            select(PlatformOrganization).where(
                PlatformOrganization.company_id == org_id
            )
        )
    elif org_type == "institution":
        org = db.session.scalar(
            select(PlatformOrganization).where(
                PlatformOrganization.institution_id == org_id
            )
        )
    else:
        return None
    return str(org.public_id) if org else None


def list_organizations_with_read_mode(**kwargs: Any) -> dict[str, Any]:
    mode = organizations_read_mode()
    if mode == "control_plane":
        return list_control_plane_organizations(**kwargs)

    # legacy (+ compare)
    legacy = list_partner_organizations(
        page=kwargs.get("page", 1),
        per_page=kwargs.get("per_page", 25),
        include_synthetic=kwargs.get("include_synthetic", False),
        organization_type=kwargs.get("organization_type"),
        configuration_status=kwargs.get("configuration_status"),
        search=kwargs.get("search"),
    )
    # Exclure orphelins de la liste « organisations »
    legacy["items"] = [
        i
        for i in legacy["items"]
        if i.get("organization_type") in ("company", "institution")
    ]
    if mode == "compare":
        for item in legacy["items"]:
            otype = item.get("organization_type")
            oid = item.get("organization_id")
            public_id = _cp_public_id_for_legacy(otype, oid)
            item["public_id"] = public_id
            item["read_source"] = "legacy"
            item["comparison_state"] = (
                "matched" if public_id else "missing_in_cp"
            )
        legacy["read_mode"] = "compare"
    else:
        legacy["read_mode"] = "legacy"
    return legacy


def get_organization_by_public_id(public_id: str) -> dict[str, Any] | None:
    try:
        uid = UUID(public_id)
    except ValueError:
        return None
    org = db.session.scalar(
        select(PlatformOrganization).where(PlatformOrganization.public_id == uid)
    )
    if org is None:
        return None
    payload = _serialize_cp_org(org)
    memberships = db.session.scalars(
        select(OrganizationMembership).where(
            OrganizationMembership.organization_id == org.id,
            OrganizationMembership.membership_status != "removed",
        )
    ).all()
    users = []
    for m in memberships:
        u = db.session.get(User, m.user_id)
        if not u:
            continue
        users.append(
            {
                "user_id": u.id,
                "name": f"{u.first_name or ''} {u.last_name or ''}".strip()
                or u.username,
                "email": u.email,
                "membership_status": m.membership_status,
                "role_template_id": m.role_template_id,
            }
        )
    anomalies = db.session.scalars(
        select(ControlPlaneAnomaly).where(
            ControlPlaneAnomaly.organization_id == org.id,
            ControlPlaneAnomaly.resolved_at.is_(None),
        )
    ).all()
    payload["users_detected"] = users
    payload["anomalies"] = [
        {
            "code": a.code,
            "severity": a.severity,
            "entity_type": a.entity_type,
            "entity_key": a.entity_key,
            "first_seen_at": a.first_seen_at.isoformat() if a.first_seen_at else None,
            "last_seen_at": a.last_seen_at.isoformat() if a.last_seen_at else None,
        }
        for a in anomalies
    ]
    payload["readiness"] = {
        "identity_ready": bool(payload.get("name") and payload.get("contact_email")),
        "access_ready": any(
            m.membership_status == "active" for m in memberships
        ),
        "services_confirmed": False,
    }
    return payload


def list_anomalies(
    *,
    page: int = 1,
    per_page: int = 50,
    entity_type: str | None = None,
    severity: str | None = None,
    code: str | None = None,
    unresolved_only: bool = True,
) -> dict[str, Any]:
    page = max(1, page)
    per_page = min(max(1, per_page), 100)
    query = select(ControlPlaneAnomaly)
    if unresolved_only:
        query = query.where(ControlPlaneAnomaly.resolved_at.is_(None))
    if entity_type:
        query = query.where(ControlPlaneAnomaly.entity_type == entity_type)
    if severity:
        query = query.where(ControlPlaneAnomaly.severity == severity)
    if code:
        query = query.where(ControlPlaneAnomaly.code == code)
    total = db.session.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.session.scalars(
        query.order_by(ControlPlaneAnomaly.last_seen_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
    ).all()
    return {
        "items": [
            {
                "id": a.id,
                "code": a.code,
                "severity": a.severity,
                "entity_type": a.entity_type,
                "entity_key": a.entity_key,
                "organization_id": a.organization_id,
                "user_id": a.user_id,
                "details": a.details_json or {},
                "first_seen_at": a.first_seen_at.isoformat() if a.first_seen_at else None,
                "last_seen_at": a.last_seen_at.isoformat() if a.last_seen_at else None,
                "resolved_at": a.resolved_at.isoformat() if a.resolved_at else None,
            }
            for a in rows
        ],
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": int(total),
            "pages": max(1, (int(total) + per_page - 1) // per_page),
        },
    }
