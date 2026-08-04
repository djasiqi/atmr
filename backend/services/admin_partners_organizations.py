"""Service lecture organisations partenaires (PR1).

Projection SQL UNION ALL paginée : companies, institutions,
comptes orphelins company/institution.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Integer, String, and_, case, cast, func, literal, or_, select, union_all
from sqlalchemy.orm import aliased

from ext import db
from models import Booking, Company, DemoAccess, Driver, Institution, User, UserRole
from models.base import _iso
from models.enums import InstitutionRole
from services.admin_role_utils import normalized_role_value

logger = logging.getLogger(__name__)


def synthetic_email_sql(column):
    """Expression SQL : e-mail synthétique / démo / interne (heuristique)."""
    lowered = func.lower(func.coalesce(column, ""))
    return or_(
        lowered.like("%@demo.local"),
        lowered.like("%@demo.lirie.ch"),
        lowered.like("demo-%@%"),
        lowered.like("%@internal.atmr.local"),
    )


def _company_branch():
    owner = aliased(User, name="owner_user")
    drivers_count_sq = (
        select(func.count(Driver.id))
        .where(Driver.company_id == Company.id)
        .correlate(Company)
        .scalar_subquery()
    )
    accounts_count_expr = case((Company.user_id.isnot(None), 1), else_=0) + (
        select(func.count(func.distinct(Driver.user_id)))
        .where(
            Driver.company_id == Company.id,
            Driver.user_id.isnot(None),
            Driver.user_id != Company.user_id,
        )
        .correlate(Company)
        .scalar_subquery()
    )

    owner_synthetic = synthetic_email_sql(owner.email)
    contact_synthetic = synthetic_email_sql(Company.contact_email)
    is_synthetic = or_(owner_synthetic, and_(owner.email.is_(None), contact_synthetic))

    return (
        select(
            literal("company").label("organization_type"),
            Company.id.label("organization_id"),
            literal(None).cast(Integer).label("account_id"),
            Company.name.label("name"),
            func.coalesce(Company.contact_email, owner.email).label("contact_email"),
            literal("complete").label("configuration_status"),
            case(
                (Company.platform_suspended.is_(True), literal("suspended")),
                else_=literal("active"),
            ).label("lifecycle_status"),
            case(
                (is_synthetic, literal("inferred_synthetic")),
                else_=literal("production"),
            ).label("data_scope"),
            literal("heuristic").label("data_scope_confidence"),
            case((is_synthetic, literal(True)), else_=literal(False)).label(
                "contains_synthetic_accounts"
            ),
            func.coalesce(Company.platform_billing_access_state, "active").label(
                "commercial_access_state"
            ),
            Company.user_id.label("primary_account_id"),
            owner.username.label("primary_account_name"),
            owner.email.label("primary_account_email"),
            cast(owner.role, String).label("primary_account_role"),
            accounts_count_expr.label("accounts_count"),
            case((Company.user_id.isnot(None), 1), else_=0).label(
                "administrators_count"
            ),
            drivers_count_sq.label("drivers_count"),
            Company.created_at.label("created_at"),
            case((is_synthetic, literal(True)), else_=literal(False)).label(
                "is_synthetic_inferred"
            ),
        )
        .select_from(Company)
        .outerjoin(owner, owner.id == Company.user_id)
    )


def _institution_branch():
    users_count_sq = (
        select(func.count(User.id))
        .where(User.institution_id == Institution.id)
        .correlate(Institution)
        .scalar_subquery()
    )
    admins_count_sq = (
        select(func.count(User.id))
        .where(
            User.institution_id == Institution.id,
            User.institution_role == InstitutionRole.ADMIN.value,
        )
        .correlate(Institution)
        .scalar_subquery()
    )
    synth_users_sq = (
        select(func.count(User.id))
        .where(
            User.institution_id == Institution.id,
            synthetic_email_sql(User.email),
        )
        .correlate(Institution)
        .scalar_subquery()
    )
    prod_users_sq = (
        select(func.count(User.id))
        .where(
            User.institution_id == Institution.id,
            ~synthetic_email_sql(User.email),
        )
        .correlate(Institution)
        .scalar_subquery()
    )

    is_synth = and_(synth_users_sq > 0, prod_users_sq == 0)
    contains_synth = synth_users_sq > 0
    config_status = case(
        (users_count_sq == 0, literal("incomplete")),
        else_=literal("complete"),
    )

    return select(
        literal("institution").label("organization_type"),
        Institution.id.label("organization_id"),
        literal(None).cast(Integer).label("account_id"),
        Institution.name.label("name"),
        Institution.contact_email.label("contact_email"),
        config_status.label("configuration_status"),
        literal("unknown").label("lifecycle_status"),
        case(
            (is_synth, literal("inferred_synthetic")),
            else_=literal("production"),
        ).label("data_scope"),
        literal("heuristic").label("data_scope_confidence"),
        contains_synth.label("contains_synthetic_accounts"),
        literal("not_applicable").label("commercial_access_state"),
        literal(None).cast(Integer).label("primary_account_id"),
        literal(None).cast(String).label("primary_account_name"),
        literal(None).cast(String).label("primary_account_email"),
        literal(None).cast(String).label("primary_account_role"),
        users_count_sq.label("accounts_count"),
        admins_count_sq.label("administrators_count"),
        literal(None).cast(Integer).label("drivers_count"),
        Institution.created_at.label("created_at"),
        case((is_synth, literal(True)), else_=literal(False)).label(
            "is_synthetic_inferred"
        ),
    ).select_from(Institution)


def _orphan_company_branch():
    has_company = (
        select(Company.id).where(Company.user_id == User.id).correlate(User).exists()
    )
    is_synth = synthetic_email_sql(User.email)
    return (
        select(
            literal("company_account_without_organization").label("organization_type"),
            literal(None).cast(Integer).label("organization_id"),
            User.id.label("account_id"),
            func.coalesce(User.username, User.email).label("name"),
            User.email.label("contact_email"),
            literal("incomplete").label("configuration_status"),
            literal("unknown").label("lifecycle_status"),
            case(
                (is_synth, literal("inferred_synthetic")),
                else_=literal("production"),
            ).label("data_scope"),
            literal("heuristic").label("data_scope_confidence"),
            case((is_synth, literal(True)), else_=literal(False)).label(
                "contains_synthetic_accounts"
            ),
            literal("not_applicable").label("commercial_access_state"),
            User.id.label("primary_account_id"),
            User.username.label("primary_account_name"),
            User.email.label("primary_account_email"),
            cast(User.role, String).label("primary_account_role"),
            literal(1).label("accounts_count"),
            literal(0).label("administrators_count"),
            literal(None).cast(Integer).label("drivers_count"),
            User.created_at.label("created_at"),
            case((is_synth, literal(True)), else_=literal(False)).label(
                "is_synthetic_inferred"
            ),
        )
        .select_from(User)
        .where(User.role == UserRole.COMPANY, ~has_company)
    )


def _orphan_institution_branch():
    is_synth = synthetic_email_sql(User.email)
    return (
        select(
            literal("institution_account_without_organization").label(
                "organization_type"
            ),
            literal(None).cast(Integer).label("organization_id"),
            User.id.label("account_id"),
            func.coalesce(User.username, User.email).label("name"),
            User.email.label("contact_email"),
            literal("incomplete").label("configuration_status"),
            literal("unknown").label("lifecycle_status"),
            case(
                (is_synth, literal("inferred_synthetic")),
                else_=literal("production"),
            ).label("data_scope"),
            literal("heuristic").label("data_scope_confidence"),
            case((is_synth, literal(True)), else_=literal(False)).label(
                "contains_synthetic_accounts"
            ),
            literal("not_applicable").label("commercial_access_state"),
            User.id.label("primary_account_id"),
            User.username.label("primary_account_name"),
            User.email.label("primary_account_email"),
            cast(User.role, String).label("primary_account_role"),
            literal(1).label("accounts_count"),
            literal(0).label("administrators_count"),
            literal(None).cast(Integer).label("drivers_count"),
            User.created_at.label("created_at"),
            case((is_synth, literal(True)), else_=literal(False)).label(
                "is_synthetic_inferred"
            ),
        )
        .select_from(User)
        .where(User.role == UserRole.INSTITUTION, User.institution_id.is_(None))
    )


def _organizations_subquery():
    return union_all(
        _company_branch(),
        _institution_branch(),
        _orphan_company_branch(),
        _orphan_institution_branch(),
    ).subquery("partners_orgs")


def _organization_key(org_type: str, org_id: int | None, account_id: int | None) -> str:
    if org_type == "company" and org_id is not None:
        return f"company:{org_id}"
    if org_type == "institution" and org_id is not None:
        return f"institution:{org_id}"
    if org_type == "company_account_without_organization" and account_id is not None:
        return f"orphan-company-account:{account_id}"
    if (
        org_type == "institution_account_without_organization"
        and account_id is not None
    ):
        return f"orphan-institution-account:{account_id}"
    return f"unknown:{org_id or account_id or 0}"


def _row_to_dict(row: Any) -> dict[str, Any]:
    org_type = row.organization_type
    org_id = row.organization_id
    account_id = row.account_id
    primary = None
    if row.primary_account_id is not None:
        primary = {
            "id": int(row.primary_account_id),
            "name": row.primary_account_name,
            "email": row.primary_account_email,
            "role": (
                normalized_role_value(row.primary_account_role)
                if row.primary_account_role
                else None
            ),
        }
    drivers_count = row.drivers_count
    if org_type == "institution":
        drivers_count = None
    return {
        "organization_key": _organization_key(org_type, org_id, account_id),
        "organization_type": org_type,
        "organization_id": int(org_id) if org_id is not None else None,
        "account_id": int(account_id) if account_id is not None else None,
        "name": row.name,
        "contact_email": row.contact_email,
        "configuration_status": row.configuration_status,
        "lifecycle_status": row.lifecycle_status,
        "data_scope": row.data_scope,
        "data_scope_confidence": row.data_scope_confidence,
        "contains_synthetic_accounts": bool(row.contains_synthetic_accounts),
        "commercial_access_state": row.commercial_access_state,
        "primary_account": primary,
        "accounts_count": int(row.accounts_count or 0),
        "administrators_count": int(row.administrators_count or 0),
        "drivers_count": int(drivers_count) if drivers_count is not None else None,
        "created_at": _iso(row.created_at) if row.created_at else None,
    }


def list_partner_organizations(
    *,
    page: int = 1,
    per_page: int = 50,
    include_synthetic: bool = False,
    organization_type: str | None = None,
    configuration_status: str | None = None,
    search: str | None = None,
    real_organizations_only: bool = False,
) -> dict[str, Any]:
    """Liste paginée des organisations partenaires."""
    page = max(page, 1)
    per_page = min(max(per_page, 1), 200)
    sq = _organizations_subquery()

    filters = []
    if not include_synthetic:
        filters.append(sq.c.data_scope != "inferred_synthetic")
    if real_organizations_only:
        filters.append(sq.c.organization_type.in_(("company", "institution")))
    if organization_type:
        filters.append(sq.c.organization_type == organization_type.strip())
    if configuration_status:
        filters.append(sq.c.configuration_status == configuration_status.strip())
    if search:
        term = f"%{search.strip().lower()}%"
        filters.append(
            or_(
                func.lower(func.coalesce(sq.c.name, "")).like(term),
                func.lower(func.coalesce(sq.c.contact_email, "")).like(term),
            )
        )

    base = select(sq)
    if filters:
        base = base.where(and_(*filters))

    total = db.session.scalar(select(func.count()).select_from(base.subquery())) or 0

    ordered = base.order_by(sq.c.created_at.desc().nullslast(), sq.c.name.asc())
    rows = db.session.execute(
        ordered.limit(per_page).offset((page - 1) * per_page)
    ).all()

    items = [_row_to_dict(r) for r in rows]
    summary = build_partners_summary(
        include_synthetic=include_synthetic,
        real_organizations_only=real_organizations_only,
    )

    return {
        "items": items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": int(total),
            "pages": max(1, (int(total) + per_page - 1) // per_page),
        },
        "summary": summary["summary"],
        "summary_scope": summary["summary_scope"],
    }


def build_partners_summary(
    *,
    include_synthetic: bool = False,
    real_organizations_only: bool = False,
) -> dict[str, Any]:
    """KPI Partenaires (définitions PR1)."""
    sq = _organizations_subquery()
    org_filters = []
    if not include_synthetic:
        org_filters.append(sq.c.data_scope != "inferred_synthetic")
    if real_organizations_only:
        org_filters.append(sq.c.organization_type.in_(("company", "institution")))

    configured_q = (
        select(func.count())
        .select_from(sq)
        .where(
            sq.c.configuration_status == "complete",
            sq.c.organization_type.in_(("company", "institution")),
            *org_filters,
        )
    )
    incomplete_q = (
        select(func.count())
        .select_from(sq)
        .where(sq.c.configuration_status == "incomplete", *org_filters)
    )
    restricted_q = (
        select(func.count())
        .select_from(sq)
        .where(
            sq.c.organization_type == "company",
            sq.c.commercial_access_state.in_(("partial", "full")),
            *org_filters,
        )
    )

    now = datetime.now(UTC)
    active_demos_q = select(func.count(func.distinct(DemoAccess.demo_request_id))).where(
        DemoAccess.status == "active",
        DemoAccess.demo_expires_at.isnot(None),
        DemoAccess.demo_expires_at > now,
    )

    return {
        "summary": {
            "configured_organizations": int(db.session.scalar(configured_q) or 0),
            "incomplete_configurations": int(db.session.scalar(incomplete_q) or 0),
            "restricted_companies": int(db.session.scalar(restricted_q) or 0),
            "active_demonstrations": int(db.session.scalar(active_demos_q) or 0),
        },
        "summary_scope": {
            "organizations_include_synthetic": include_synthetic,
            "demonstrations_include_all": True,
        },
    }


def build_account_integrity(user_id: int) -> dict[str, Any] | None:
    """Diagnostic d'intégrité lecture seule pour un compte."""
    user = db.session.get(User, user_id)
    if user is None:
        return None

    role = normalized_role_value(user.role)
    company = db.session.scalars(
        select(Company).where(Company.user_id == user.id)
    ).first()
    driver = getattr(user, "driver", None)
    clients_count = len(getattr(user, "clients", None) or [])
    bookings_created = (
        db.session.scalar(
            select(func.count(Booking.id)).where(Booking.user_id == user.id)
        )
        or 0
    )

    from models.refresh_token import RefreshToken

    refresh_sessions = (
        db.session.scalar(
            select(func.count(RefreshToken.id)).where(RefreshToken.user_id == user.id)
        )
        or 0
    )

    exact_conflicts: list[dict[str, Any]] = []
    possible_matches: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    if role == UserRole.COMPANY.value:
        if company is None:
            checks.append(
                {
                    "code": "COMPANY_PROFILE_LINKED",
                    "status": "failed",
                    "severity": "blocking",
                    "label": "Fiche entreprise liée",
                }
            )
            email = (user.email or "").strip().lower()
            if email:
                matches = db.session.scalars(
                    select(Company).where(
                        func.lower(func.coalesce(Company.contact_email, "")) == email
                    )
                ).all()
                for match in matches:
                    possible_matches.append(
                        {
                            "organization_key": f"company:{match.id}",
                            "reason": "same_normalized_email",
                            "confidence": "high",
                        }
                    )
                if matches:
                    checks.append(
                        {
                            "code": "POSSIBLE_COMPANY_MATCH",
                            "status": "warning",
                            "severity": "warning",
                            "label": (
                                "Une entreprise utilisant le même e-mail "
                                "existe peut-être"
                            ),
                        }
                    )
        else:
            checks.append(
                {
                    "code": "COMPANY_PROFILE_LINKED",
                    "status": "passed",
                    "severity": "blocking",
                    "label": "Fiche entreprise liée",
                }
            )
            siblings = db.session.scalars(
                select(Company).where(
                    Company.user_id == user.id, Company.id != company.id
                )
            ).all()
            for sibling in siblings:
                exact_conflicts.append(
                    {
                        "organization_key": f"company:{sibling.id}",
                        "reason": "same_user_id",
                        "confidence": "certain",
                    }
                )

    if role == UserRole.INSTITUTION.value:
        if user.institution_id is None:
            checks.append(
                {
                    "code": "INSTITUTION_LINKED",
                    "status": "failed",
                    "severity": "blocking",
                    "label": "Institution liée",
                }
            )
        else:
            checks.append(
                {
                    "code": "INSTITUTION_LINKED",
                    "status": "passed",
                    "severity": "blocking",
                    "label": "Institution liée",
                }
            )

    checks.append(
        {
            "code": "USER_ACCOUNT_ACTIVE",
            "status": "passed",
            "severity": "info",
            "label": "Compte utilisateur",
        }
    )

    incomplete = any(
        c["status"] == "failed" and c["severity"] == "blocking" for c in checks
    )

    return {
        "account": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": role,
            "created_at": _iso(user.created_at) if user.created_at else None,
            "institution_id": user.institution_id,
            "company_id": company.id if company else None,
        },
        "configuration_status": "incomplete" if incomplete else "complete",
        "exact_conflicts": exact_conflicts,
        "possible_matches": possible_matches,
        "dependencies": {
            "driver_profile_exists": driver is not None,
            "institution_link_exists": user.institution_id is not None,
            "client_profiles_count": clients_count,
            "bookings_created_by_account_count": int(bookings_created),
            "refresh_sessions_count": int(refresh_sessions),
        },
        "checks": checks,
        "recommendation": (
            "Vérifier l'existence d'une organisation portant le même e-mail "
            "avant de créer ou rattacher une nouvelle fiche."
            if incomplete
            else None
        ),
    }
