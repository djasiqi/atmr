"""Gate de cutover lecture control_plane (CP-PR1)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from ext import db
from models.company import Company
from models.control_plane import (
    ControlPlaneAnomaly,
    OrganizationMembership,
    PlatformOrganization,
)
from models.enums import UserRole
from models.institution import Institution
from models.user import User
from services.control_plane.classification import (
    CompanyProjectionKind,
    classify_company_for_control_plane,
)


def control_plane_cutover_status() -> dict[str, Any]:
    """Statut readiness pour activer CONTROL_PLANE_ORGANIZATIONS_READ_MODE=control_plane."""
    blockers: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    companies = db.session.scalars(select(Company)).all()
    missing_projections = 0
    ambiguous_companies = 0
    for company in companies:
        decision = classify_company_for_control_plane(company)
        if decision.kind == CompanyProjectionKind.AMBIGUOUS:
            ambiguous_companies += 1
            continue
        if decision.kind != CompanyProjectionKind.TRANSPORT_TENANT:
            continue
        org = db.session.scalar(
            select(PlatformOrganization).where(
                PlatformOrganization.company_id == company.id
            )
        )
        if org is None:
            missing_projections += 1

    institutions = db.session.scalars(select(Institution)).all()
    for inst in institutions:
        org = db.session.scalar(
            select(PlatformOrganization).where(
                PlatformOrganization.institution_id == inst.id
            )
        )
        if org is None:
            missing_projections += 1

    if missing_projections:
        blockers.append(
            {
                "code": "MISSING_PROJECTIONS",
                "count": missing_projections,
                "message": "Organisations legacy attendues absentes du control plane",
            }
        )

    if ambiguous_companies:
        blockers.append(
            {
                "code": "AMBIGUOUS_COMPANIES",
                "count": ambiguous_companies,
                "message": "Companies ambiguës non résolues (override ou reconcile requis)",
            }
        )

    unknown_origins = (
        db.session.scalar(
            select(func.count())
            .select_from(PlatformOrganization)
            .where(PlatformOrganization.data_origin == "unknown")
        )
        or 0
    )
    if unknown_origins:
        warnings.append(
            {
                "code": "UNKNOWN_ORIGINS",
                "count": int(unknown_origins),
                "message": "Organisations CP avec data_origin=unknown",
            }
        )

    critical_anomalies = (
        db.session.scalar(
            select(func.count())
            .select_from(ControlPlaneAnomaly)
            .where(
                ControlPlaneAnomaly.resolved_at.is_(None),
                ControlPlaneAnomaly.severity == "critical",
            )
        )
        or 0
    )
    if critical_anomalies:
        blockers.append(
            {
                "code": "CRITICAL_ANOMALIES",
                "count": int(critical_anomalies),
                "message": "Anomalies critiques non résolues",
            }
        )

    # Memberships institution divergentes : users institution sans membership active CP
    inst_users = db.session.scalars(
        select(User).where(
            User.role == UserRole.INSTITUTION,
            User.institution_id.isnot(None),
            User.archived_at.is_(None),
        )
    ).all()
    divergent_memberships = 0
    for u in inst_users:
        org = db.session.scalar(
            select(PlatformOrganization).where(
                PlatformOrganization.institution_id == u.institution_id
            )
        )
        if org is None:
            divergent_memberships += 1
            continue
        m = db.session.scalar(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == org.id,
                OrganizationMembership.user_id == u.id,
                OrganizationMembership.membership_status != "removed",
            )
        )
        if m is None:
            divergent_memberships += 1

    if divergent_memberships:
        blockers.append(
            {
                "code": "DIVERGENT_MEMBERSHIPS",
                "count": divergent_memberships,
                "message": "Utilisateurs institution sans membership CP alignée",
            }
        )

    ready = len(blockers) == 0
    return {
        "ready": ready,
        "blockers": blockers,
        "warnings": warnings,
        "stats": {
            "companies_scanned": len(companies),
            "institutions_scanned": len(institutions),
            "ambiguous_companies": ambiguous_companies,
            "missing_projections": missing_projections,
            "unknown_origins": int(unknown_origins),
            "critical_anomalies": int(critical_anomalies),
            "divergent_memberships": divergent_memberships,
        },
    }


def assert_control_plane_read_cutover_ready() -> None:
    """Lève RuntimeError si le cutover lecture control_plane n'est pas prêt."""
    status = control_plane_cutover_status()
    if status["ready"]:
        return
    codes = ", ".join(b["code"] for b in status["blockers"])
    raise RuntimeError(
        f"Cutover control_plane non prêt : {codes}. "
        "Exécuter `flask control-plane cutover-status` pour le détail."
    )
