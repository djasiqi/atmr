"""Réconciliation + anomalies persistées (CP-PR1)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from ext import db
from models.company import Company
from models.control_plane import (
    ControlPlaneAnomaly,
    OrganizationMembership,
    PlatformOrganization,
    RoleTemplate,
)
from models.driver import Driver
from models.enums import InstitutionRole, UserRole
from models.institution import Institution
from models.user import User
from services.control_plane.classification import (
    CompanyProjectionKind,
    classify_company_for_control_plane,
)
from services.control_plane.projector import get_projector
from services.control_plane.seed import seed_control_plane_catalogs

logger = logging.getLogger(__name__)


def _fingerprint(code: str, entity_type: str, entity_key: str) -> str:
    raw = f"{code}|{entity_type}|{entity_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:64]


def _upsert_anomaly(
    *,
    code: str,
    severity: str,
    entity_type: str,
    entity_key: str,
    organization_id: int | None = None,
    user_id: int | None = None,
    details: dict[str, Any] | None = None,
    seen: set[str],
) -> None:
    fp = _fingerprint(code, entity_type, entity_key)
    seen.add(fp)
    now = datetime.now(UTC)
    existing = db.session.scalar(
        select(ControlPlaneAnomaly).where(ControlPlaneAnomaly.fingerprint == fp)
    )
    if existing is None:
        db.session.add(
            ControlPlaneAnomaly(
                fingerprint=fp,
                code=code,
                severity=severity,
                entity_type=entity_type,
                entity_key=entity_key,
                organization_id=organization_id,
                user_id=user_id,
                details_json=details or {},
                first_seen_at=now,
                last_seen_at=now,
                resolved_at=None,
            )
        )
        return
    existing.last_seen_at = now
    existing.severity = severity
    existing.details_json = details or {}
    existing.organization_id = organization_id
    existing.user_id = user_id
    if existing.resolved_at is not None:
        existing.resolved_at = None
        existing.resolution_source = None


def reconcile_control_plane(*, dry_run: bool = True, apply_projection: bool = True) -> dict[str, Any]:
    """Recalcule projections (si apply) et anomalies. dry_run = rollback final."""
    seed_control_plane_catalogs(commit=False)
    projector = get_projector()
    seen: set[str] = set()
    stats = {
        "companies_scanned": 0,
        "tenants_projected": 0,
        "shells": 0,
        "ambiguous": 0,
        "institutions_projected": 0,
        "drivers_synced": 0,
        "institution_users_synced": 0,
        "anomalies_open": 0,
    }

    # --- Companies ---
    companies = db.session.scalars(select(Company)).all()
    owner_tenant_counts: dict[int, list[int]] = {}
    for company in companies:
        stats["companies_scanned"] += 1
        decision = classify_company_for_control_plane(company)
        if decision.kind == CompanyProjectionKind.BILLING_SHELL:
            stats["shells"] += 1
            _upsert_anomaly(
                code="COMPANY_BILLING_SHELL_EXCLUDED",
                severity="info",
                entity_type="organization",
                entity_key=f"company:{company.id}",
                details={"reason": decision.reason, **decision.evidence},
                seen=seen,
            )
            continue
        if decision.kind == CompanyProjectionKind.AMBIGUOUS:
            stats["ambiguous"] += 1
            _upsert_anomaly(
                code="COMPANY_TENANT_CLASSIFICATION_AMBIGUOUS",
                severity="critical",
                entity_type="organization",
                entity_key=f"company:{company.id}",
                details={"reason": decision.reason, **decision.evidence},
                seen=seen,
            )
            continue

        # TRANSPORT_TENANT
        if apply_projection:
            org = projector.ensure_company_organization(company)
            if org is not None:
                stats["tenants_projected"] += 1
                owner_tenant_counts.setdefault(int(company.user_id), []).append(
                    int(company.id)
                )

    for owner_id, company_ids in owner_tenant_counts.items():
        if len(company_ids) > 1:
            _upsert_anomaly(
                code="DUPLICATE_TENANT_COMPANY_OWNER",
                severity="critical",
                entity_type="account",
                entity_key=f"user:{owner_id}",
                user_id=owner_id,
                details={"company_ids": company_ids},
                seen=seen,
            )

    # --- Institutions ---
    institutions = db.session.scalars(select(Institution)).all()
    for inst in institutions:
        if apply_projection:
            org = projector.ensure_institution_organization(inst)
            projector.ensure_shadow_entitlements_institution(org)
            stats["institutions_projected"] += 1
            users = db.session.scalars(
                select(User).where(User.institution_id == inst.id)
            ).all()
            admin_count = 0
            for u in users:
                projector.sync_institution_user(u)
                stats["institution_users_synced"] += 1
                if (
                    u.institution_role == InstitutionRole.ADMIN.value
                    and u.archived_at is None
                    and u.disabled_at is None
                    and (u.account_status is None or u.account_status == "active")
                ):
                    admin_count += 1
            if admin_count == 0:
                _upsert_anomaly(
                    code="ORGANIZATION_WITHOUT_ACTIVE_ADMIN",
                    severity="warning",
                    entity_type="organization",
                    entity_key=f"institution:{inst.id}",
                    organization_id=org.id if org else None,
                    details={"institution_id": inst.id},
                    seen=seen,
                )

    # --- Drivers ---
    drivers = db.session.scalars(select(Driver)).all()
    for driver in drivers:
        user = db.session.get(User, driver.user_id)
        if user is None:
            _upsert_anomaly(
                code="DRIVER_ROLE_MISMATCH",
                severity="warning",
                entity_type="account",
                entity_key=f"driver:{driver.id}",
                details={"reason": "driver_user_missing"},
                seen=seen,
            )
            continue
        role_val = getattr(user.role, "value", user.role)
        if str(role_val).upper() != "DRIVER":
            _upsert_anomaly(
                code="DRIVER_ROLE_MISMATCH",
                severity="warning",
                entity_type="account",
                entity_key=f"user:{user.id}",
                user_id=int(user.id),
                details={"role": str(role_val), "driver_id": driver.id},
                seen=seen,
            )
        if apply_projection:
            projector.sync_driver(driver)
            stats["drivers_synced"] += 1

    # --- Orphan accounts ---
    company_users = db.session.scalars(
        select(User).where(User.role == UserRole.COMPANY)
    ).all()
    for u in company_users:
        has_company = db.session.scalar(
            select(Company.id).where(Company.user_id == u.id).limit(1)
        )
        if has_company is None:
            _upsert_anomaly(
                code="ACCOUNT_COMPANY_PROFILE_MISSING",
                severity="warning",
                entity_type="account",
                entity_key=f"user:{u.id}",
                user_id=int(u.id),
                details={},
                seen=seen,
            )

    inst_users = db.session.scalars(
        select(User).where(User.role == UserRole.INSTITUTION)
    ).all()
    for u in inst_users:
        if u.institution_id is None:
            _upsert_anomaly(
                code="ACCOUNT_INSTITUTION_LINK_MISSING",
                severity="warning",
                entity_type="account",
                entity_key=f"user:{u.id}",
                user_id=int(u.id),
                details={},
                seen=seen,
            )

    driver_role_users = db.session.scalars(
        select(User).where(User.role == UserRole.DRIVER)
    ).all()
    for u in driver_role_users:
        has_driver = db.session.scalar(
            select(Driver.id).where(Driver.user_id == u.id).limit(1)
        )
        if has_driver is None:
            _upsert_anomaly(
                code="ACCOUNT_DRIVER_PROFILE_MISSING",
                severity="warning",
                entity_type="account",
                entity_key=f"user:{u.id}",
                user_id=int(u.id),
                details={},
                seen=seen,
            )

    # DATA_ORIGIN_UNKNOWN (échantillon org projetées)
    unknown_orgs = db.session.scalars(
        select(PlatformOrganization).where(
            PlatformOrganization.data_origin == "unknown"
        )
    ).all()
    for org in unknown_orgs:
        _upsert_anomaly(
            code="DATA_ORIGIN_UNKNOWN",
            severity="info",
            entity_type="organization",
            entity_key=f"platform_organization:{org.id}",
            organization_id=int(org.id),
            details={"public_id": str(org.public_id)},
            seen=seen,
        )

    # Résoudre anomalies disparues
    open_anomalies = db.session.scalars(
        select(ControlPlaneAnomaly).where(ControlPlaneAnomaly.resolved_at.is_(None))
    ).all()
    now = datetime.now(UTC)
    resolved = 0
    for a in open_anomalies:
        if a.fingerprint not in seen:
            a.resolved_at = now
            a.resolution_source = "reconcile"
            resolved += 1

    stats["anomalies_open"] = len(seen)
    stats["anomalies_resolved"] = resolved

    if dry_run:
        db.session.rollback()
        logger.info("[cp.reconcile] dry-run stats=%s", stats)
    else:
        db.session.commit()
        logger.info("[cp.reconcile] apply stats=%s", stats)

    return stats
