"""Projecteur transactionnel Legacy → Control plane (CP-PR1)."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ext import db
from models.company import Company
from models.control_plane import (
    OrganizationMembership,
    OrganizationServiceEntitlement,
    PlatformOrganization,
    RoleTemplate,
    ServiceCatalog,
)
from models.driver import Driver
from models.enums import InstitutionRole
from models.institution import Institution
from models.user import User
from services.control_plane.classification import (
    CompanyProjectionKind,
    apply_user_classification,
    classify_company_for_control_plane,
    classify_organization_data_origin,
    classify_user_data_origin,
    derive_company_lifecycle,
    derive_institution_lifecycle,
)

logger = logging.getLogger(__name__)

_INSTITUTION_ROLE_KEYS = {r.value for r in InstitutionRole}


def _now() -> datetime:
    return datetime.now(UTC)


def _role_template_id(organization_type: str, role_key: str) -> int | None:
    row = db.session.scalar(
        select(RoleTemplate.id).where(
            RoleTemplate.organization_type == organization_type,
            RoleTemplate.role_key == role_key,
        )
    )
    return int(row) if row is not None else None


def membership_status_from_user(user: User) -> str:
    if getattr(user, "archived_at", None) is not None:
        return "removed"
    if getattr(user, "disabled_at", None) is not None:
        return "suspended"
    status = getattr(user, "account_status", None)
    if status == "disabled":
        return "suspended"
    if status in ("invited", "pending_activation"):
        return "invited"
    return "active"


class ControlPlaneProjector:
    """Projection unidirectionnelle Legacy → CP, upsert concurrent-safe."""

    def ensure_company_organization(
        self, company: Company
    ) -> PlatformOrganization | None:
        decision = classify_company_for_control_plane(company)
        if decision.kind != CompanyProjectionKind.TRANSPORT_TENANT:
            logger.debug(
                "[cp.projector] skip company id=%s kind=%s reason=%s",
                company.id,
                decision.kind,
                decision.reason,
            )
            return None

        origin = classify_organization_data_origin(
            organization_type="company", company=company
        )
        lifecycle = derive_company_lifecycle(company)

        existing = db.session.scalar(
            select(PlatformOrganization).where(
                PlatformOrganization.company_id == company.id
            )
        )
        if existing is None:
            stmt = (
                pg_insert(PlatformOrganization)
                .values(
                    public_id=uuid.uuid4(),
                    organization_type="company",
                    company_id=company.id,
                    institution_id=None,
                    lifecycle_status=lifecycle,
                    lifecycle_source="legacy_derived",
                    data_origin=origin.data_origin,
                    data_origin_source=origin.source,
                    data_origin_confidence=origin.confidence,
                    classification_evidence_json=origin.evidence,
                    classified_at=_now(),
                    activated_at=_now() if lifecycle == "active" else None,
                    suspended_at=_now() if lifecycle == "suspended" else None,
                    created_at=_now(),
                    updated_at=_now(),
                )
                .on_conflict_do_update(
                    index_elements=["company_id"],
                    set_={
                        "updated_at": _now(),
                    },
                )
                .returning(PlatformOrganization.id)
            )
            org_id = db.session.execute(stmt).scalar_one()
            org = db.session.get(PlatformOrganization, org_id)
        else:
            org = existing
            self._maybe_update_lifecycle(org, lifecycle)
            if org.data_origin_source != "explicit_admin":
                org.data_origin = origin.data_origin
                org.data_origin_source = origin.source
                org.data_origin_confidence = origin.confidence
                org.classification_evidence_json = origin.evidence
                org.classified_at = _now()
            org.updated_at = _now()

        assert org is not None
        self.sync_company_owner(company, organization=org)
        self._ensure_shadow_entitlements_company(org, company)
        return org

    def ensure_institution_organization(
        self, institution: Institution
    ) -> PlatformOrganization:
        member_ids = list(
            db.session.scalars(
                select(User.id).where(User.institution_id == institution.id)
            ).all()
        )
        origin = classify_organization_data_origin(
            organization_type="institution",
            institution=institution,
            member_user_ids=[int(i) for i in member_ids],
        )
        lifecycle = derive_institution_lifecycle(int(institution.id))

        existing = db.session.scalar(
            select(PlatformOrganization).where(
                PlatformOrganization.institution_id == institution.id
            )
        )
        if existing is None:
            stmt = (
                pg_insert(PlatformOrganization)
                .values(
                    public_id=uuid.uuid4(),
                    organization_type="institution",
                    company_id=None,
                    institution_id=institution.id,
                    lifecycle_status=lifecycle,
                    lifecycle_source="legacy_derived",
                    data_origin=origin.data_origin,
                    data_origin_source=origin.source,
                    data_origin_confidence=origin.confidence,
                    classification_evidence_json=origin.evidence,
                    classified_at=_now(),
                    activated_at=_now() if lifecycle == "active" else None,
                    created_at=_now(),
                    updated_at=_now(),
                )
                .on_conflict_do_update(
                    index_elements=["institution_id"],
                    set_={"updated_at": _now()},
                )
                .returning(PlatformOrganization.id)
            )
            org_id = db.session.execute(stmt).scalar_one()
            org = db.session.get(PlatformOrganization, org_id)
        else:
            org = existing
            self._maybe_update_lifecycle(org, lifecycle)
            if org.data_origin_source != "explicit_admin":
                org.data_origin = origin.data_origin
                org.data_origin_source = origin.source
                org.data_origin_confidence = origin.confidence
                org.classification_evidence_json = origin.evidence
                org.classified_at = _now()
            org.updated_at = _now()

        assert org is not None
        return org

    def sync_company_owner(
        self,
        company: Company,
        *,
        organization: PlatformOrganization | None = None,
    ) -> OrganizationMembership | None:
        org = organization or db.session.scalar(
            select(PlatformOrganization).where(
                PlatformOrganization.company_id == company.id
            )
        )
        if org is None:
            return None
        user = db.session.get(User, company.user_id)
        if user is None:
            return None
        apply_user_classification(user, classify_user_data_origin(user))
        role_id = _role_template_id("company", "company_owner")
        return self._upsert_membership(
            organization_id=int(org.id),
            user_id=int(user.id),
            role_template_id=role_id,
            membership_status=membership_status_from_user(user),
            source="legacy_sync",
        )

    def sync_institution_user(self, user: User) -> OrganizationMembership | None:
        if not user.institution_id:
            return None
        institution = db.session.get(Institution, user.institution_id)
        if institution is None:
            return None
        org = self.ensure_institution_organization(institution)
        apply_user_classification(user, classify_user_data_origin(user))

        role_key = user.institution_role
        if not role_key or role_key not in _INSTITUTION_ROLE_KEYS:
            role_key = "legacy_unresolved"
            status = "needs_review"
        else:
            status = membership_status_from_user(user)

        role_id = _role_template_id("institution", role_key)
        return self._upsert_membership(
            organization_id=int(org.id),
            user_id=int(user.id),
            role_template_id=role_id,
            membership_status=status,
            source="legacy_sync",
            scope_type=(
                "curator_assignments"
                if role_key == InstitutionRole.CURATOR.value
                else "organization"
            ),
            scope_json=(
                {"curator_user_id": user.id}
                if role_key == InstitutionRole.CURATOR.value
                else {}
            ),
        )

    def sync_driver(self, driver: Driver) -> OrganizationMembership | None:
        company = db.session.get(Company, driver.company_id)
        if company is None:
            return None
        org = self.ensure_company_organization(company)
        if org is None:
            return None

        user = db.session.get(User, driver.user_id)
        if user is None:
            return None
        apply_user_classification(user, classify_user_data_origin(user))

        # Retirer les anciennes memberships company_driver sur d'autres orgs
        driver_role_id = _role_template_id("company", "company_driver")
        if driver_role_id is not None:
            others = db.session.scalars(
                select(OrganizationMembership).where(
                    OrganizationMembership.user_id == user.id,
                    OrganizationMembership.role_template_id == driver_role_id,
                    OrganizationMembership.organization_id != org.id,
                    OrganizationMembership.membership_status != "removed",
                )
            ).all()
            for m in others:
                m.membership_status = "removed"
                m.removed_at = _now()
                m.updated_at = _now()

        status = membership_status_from_user(user)
        return self._upsert_membership(
            organization_id=int(org.id),
            user_id=int(user.id),
            role_template_id=driver_role_id,
            membership_status=status,
            source="legacy_sync",
        )

    def sync_user_account_state(self, user: User) -> None:
        """Propage disable/archive vers les memberships projetées."""
        status = membership_status_from_user(user)
        memberships = db.session.scalars(
            select(OrganizationMembership).where(
                OrganizationMembership.user_id == user.id,
                OrganizationMembership.membership_status != "removed",
            )
        ).all()
        for m in memberships:
            if status == "removed":
                m.membership_status = "removed"
                m.removed_at = _now()
            elif status == "suspended":
                m.membership_status = "suspended"
                m.suspended_at = _now()
            elif m.membership_status in ("suspended", "invited") and status == "active":
                m.membership_status = "active"
                m.activated_at = m.activated_at or _now()
            m.updated_at = _now()

    def _maybe_update_lifecycle(self, org: PlatformOrganization, lifecycle: str) -> None:
        if org.lifecycle_source == "explicit_admin":
            return
        org.lifecycle_status = lifecycle
        org.lifecycle_source = "legacy_derived"
        if lifecycle == "active" and org.activated_at is None:
            org.activated_at = _now()
        if lifecycle == "suspended":
            org.suspended_at = org.suspended_at or _now()

    def _upsert_membership(
        self,
        *,
        organization_id: int,
        user_id: int,
        role_template_id: int | None,
        membership_status: str,
        source: str,
        scope_type: str = "organization",
        scope_json: dict[str, Any] | None = None,
    ) -> OrganizationMembership:
        existing = db.session.scalar(
            select(OrganizationMembership).where(
                OrganizationMembership.organization_id == organization_id,
                OrganizationMembership.user_id == user_id,
            )
        )
        if existing is None:
            stmt = (
                pg_insert(OrganizationMembership)
                .values(
                    organization_id=organization_id,
                    user_id=user_id,
                    role_template_id=role_template_id,
                    membership_status=membership_status,
                    scope_type=scope_type,
                    scope_schema_version=1,
                    scope_json=scope_json or {},
                    source=source,
                    activated_at=_now() if membership_status == "active" else None,
                    invited_at=_now() if membership_status == "invited" else None,
                    created_at=_now(),
                    updated_at=_now(),
                )
                .on_conflict_do_update(
                    constraint="uq_organization_membership_org_user",
                    set_={
                        "role_template_id": role_template_id,
                        "membership_status": membership_status,
                        "scope_type": scope_type,
                        "scope_json": scope_json or {},
                        "source": source,
                        "updated_at": _now(),
                    },
                )
                .returning(OrganizationMembership.id)
            )
            mid = db.session.execute(stmt).scalar_one()
            membership = db.session.get(OrganizationMembership, mid)
            assert membership is not None
            return membership

        existing.role_template_id = role_template_id
        existing.membership_status = membership_status
        existing.scope_type = scope_type
        existing.scope_json = scope_json or {}
        existing.source = source
        existing.updated_at = _now()
        return existing

    def _ensure_shadow_entitlements_company(
        self, org: PlatformOrganization, company: Company
    ) -> None:
        keys = ["company.own_portfolio", "company.driver_management"]
        if bool(getattr(company, "is_partner", False)) or bool(
            getattr(company, "is_approved", False)
        ):
            keys.append("company.marketplace")
        if bool(getattr(company, "dispatch_enabled", False)):
            keys.append("company.dispatch")
        keys.append("company.live_tracking")
        self._upsert_shadow_entitlements(org, keys, source="legacy_inferred")

    def ensure_shadow_entitlements_institution(self, org: PlatformOrganization) -> None:
        keys = [
            "institution.transport_coordination",
            "institution.patient_management",
            "institution.users_teams",
        ]
        self._upsert_shadow_entitlements(org, keys, source="legacy_inferred")

    def _upsert_shadow_entitlements(
        self, org: PlatformOrganization, service_keys: list[str], *, source: str
    ) -> None:
        for key in service_keys:
            svc = db.session.scalar(
                select(ServiceCatalog).where(ServiceCatalog.service_key == key)
            )
            if svc is None:
                continue
            existing = db.session.scalar(
                select(OrganizationServiceEntitlement).where(
                    OrganizationServiceEntitlement.organization_id == org.id,
                    OrganizationServiceEntitlement.service_catalog_id == svc.id,
                )
            )
            if existing is not None:
                # Ne jamais promouvoir en enforced ; ne pas écraser explicit_admin
                if existing.source == "explicit_admin":
                    continue
                if existing.enforcement_mode != "shadow":
                    existing.enforcement_mode = "shadow"
                continue
            stmt = (
                pg_insert(OrganizationServiceEntitlement)
                .values(
                    organization_id=org.id,
                    service_catalog_id=svc.id,
                    status="enabled",
                    source=source,
                    enforcement_mode="shadow",
                    confidence="heuristic",
                    reason="cp_pr1_backfill_inferred",
                    created_at=_now(),
                    updated_at=_now(),
                )
                .on_conflict_do_nothing(
                    constraint="uq_org_service_entitlement_org_service"
                )
            )
            db.session.execute(stmt)


_projector: ControlPlaneProjector | None = None


def get_projector() -> ControlPlaneProjector:
    global _projector
    if _projector is None:
        _projector = ControlPlaneProjector()
    return _projector
