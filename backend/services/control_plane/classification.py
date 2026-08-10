"""Classification Company fail-closed + lifecycle + data_origin (CP-PR1)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import func, or_, select

from ext import db
from models.billing_party import BillingParty
from models.clinic_billing_party_mapping import ClinicBillingPartyMapping
from models.company import Company
from models.control_plane import ControlPlaneEntityOverride
from models.demo_access import DemoAccess
from models.driver import Driver
from models.enums import BillingPartyType, UserRole
from models.institution import Institution
from models.user import User


class CompanyProjectionKind(StrEnum):
    TRANSPORT_TENANT = "transport_tenant"
    BILLING_SHELL = "billing_shell"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class CompanyProjectionDecision:
    kind: CompanyProjectionKind
    reason: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class DataOriginDecision:
    data_origin: str
    source: str
    confidence: str
    evidence: dict[str, Any]


_TECHNICAL_EMAIL_SUFFIXES = (
    "@demo.local",
    "@demo.lirie.ch",
    "@internal.atmr.local",
    "@test.example",
    "@example.com",
)


def _driver_count(company_id: int) -> int:
    return (
        db.session.scalar(
            select(func.count())
            .select_from(Driver)
            .where(Driver.company_id == company_id)
        )
        or 0
    )


def _is_clinic_referenced(company_id: int) -> bool:
    mapping = db.session.scalar(
        select(ClinicBillingPartyMapping.id)
        .where(ClinicBillingPartyMapping.clinic_company_id == company_id)
        .limit(1)
    )
    if mapping is not None:
        return True
    ref = f"clinic_company:{company_id}"
    bp = db.session.scalar(
        select(BillingParty.id)
        .where(
            BillingParty.external_ref == ref,
            BillingParty.type == BillingPartyType.CLINIC,
        )
        .limit(1)
    )
    return bp is not None


def _override_value(entity_type: str, entity_id: int, override_key: str) -> str | None:
    row = db.session.scalar(
        select(ControlPlaneEntityOverride).where(
            ControlPlaneEntityOverride.entity_type == entity_type,
            ControlPlaneEntityOverride.entity_id == entity_id,
            ControlPlaneEntityOverride.override_key == override_key,
        )
    )
    return row.override_value if row else None


def classify_company_for_control_plane(company: Company) -> CompanyProjectionDecision:
    """Ordre fail-closed : override → drivers → shell (owner non-COMPANY) → unique tenant → ambiguous."""
    override = _override_value("company", int(company.id), "company_projection_kind")
    if override in {k.value for k in CompanyProjectionKind}:
        return CompanyProjectionDecision(
            kind=CompanyProjectionKind(override),
            reason="explicit_override",
            evidence={"override": override},
        )

    drivers = _driver_count(int(company.id))
    if drivers > 0:
        return CompanyProjectionDecision(
            kind=CompanyProjectionKind.TRANSPORT_TENANT,
            reason="has_drivers",
            evidence={"drivers_count": drivers},
        )

    owner = db.session.get(User, company.user_id) if company.user_id else None
    owner_role = getattr(owner, "role", None) if owner else None
    owner_is_company = owner_role in (
        UserRole.COMPANY,
        UserRole.COMPANY.value,
        "COMPANY",
        "company",
    )

    clinic_ref = _is_clinic_referenced(int(company.id))
    # Shell certain uniquement si clinic + 0 driver + owner NON-COMPANY
    if clinic_ref and drivers == 0 and not owner_is_company:
        return CompanyProjectionDecision(
            kind=CompanyProjectionKind.BILLING_SHELL,
            reason="clinic_reference_non_company_owner",
            evidence={
                "clinic_referenced": True,
                "owner_role": str(owner_role) if owner_role is not None else None,
            },
        )

    # Clinic + owner COMPANY + 0 driver → ambiguous (ne pas exclure un tenant silencieux)
    if clinic_ref and drivers == 0 and owner_is_company:
        return CompanyProjectionDecision(
            kind=CompanyProjectionKind.AMBIGUOUS,
            reason="clinic_reference_company_owner_no_drivers",
            evidence={
                "clinic_referenced": True,
                "owner_id": owner.id if owner else None,
            },
        )

    if owner_is_company and owner is not None:
        siblings = list(
            db.session.scalars(select(Company).where(Company.user_id == owner.id)).all()
        )
        # Candidats non-shell : pas de clinic_ref (owner COMPANY clinic = ambiguous ailleurs)
        unique_tenants = []
        for c in siblings:
            if _driver_count(int(c.id)) > 0:
                continue
            if _is_clinic_referenced(int(c.id)):
                continue
            unique_tenants.append(c)
        if len(unique_tenants) == 1 and int(unique_tenants[0].id) == int(company.id):
            return CompanyProjectionDecision(
                kind=CompanyProjectionKind.TRANSPORT_TENANT,
                reason="unique_company_owner_candidate",
                evidence={
                    "owner_id": owner.id,
                    "sibling_count": len(siblings),
                },
            )

    return CompanyProjectionDecision(
        kind=CompanyProjectionKind.AMBIGUOUS,
        reason="unresolved_company_kind",
        evidence={
            "drivers_count": drivers,
            "clinic_referenced": clinic_ref,
            "owner_role": str(owner_role) if owner_role is not None else None,
        },
    )


def derive_company_lifecycle(company: Company) -> str:
    if bool(getattr(company, "platform_suspended", False)):
        return "suspended"
    if bool(getattr(company, "is_approved", False)) or getattr(
        company, "accepted_at", None
    ):
        return "active"
    return "onboarding"


def derive_institution_lifecycle(institution_id: int) -> str:
    """Active si ≥1 membre actif (legacy User), sinon onboarding."""
    active_count = (
        db.session.scalar(
            select(func.count())
            .select_from(User)
            .where(
                User.institution_id == institution_id,
                User.archived_at.is_(None),
                User.disabled_at.is_(None),
                or_(
                    User.account_status.is_(None),
                    User.account_status == "active",
                ),
            )
        )
        or 0
    )
    return "active" if active_count > 0 else "onboarding"


def _email_heuristic_origin(email: str | None) -> DataOriginDecision | None:
    if not email:
        return None
    lowered = email.strip().lower()
    if lowered.startswith("demo-") or any(
        lowered.endswith(sfx) for sfx in ("@demo.local", "@demo.lirie.ch")
    ):
        return DataOriginDecision(
            "demo", "email_heuristic", "heuristic", {"email": lowered}
        )
    if lowered.endswith("@internal.atmr.local"):
        return DataOriginDecision(
            "internal", "email_heuristic", "heuristic", {"email": lowered}
        )
    if (
        any(lowered.endswith(sfx) for sfx in ("@test.example", "@example.com"))
        or "testuser" in lowered
    ):
        return DataOriginDecision(
            "test", "email_heuristic", "heuristic", {"email": lowered}
        )
    return None


def classify_user_data_origin(user: User) -> DataOriginDecision:
    """Priorité : override → DemoAccess → heuristique → unknown (jamais production auto)."""
    override = _override_value("user", int(user.id), "data_origin")
    if override:
        return DataOriginDecision(
            override, "explicit_admin", "explicit", {"override": override}
        )

    demo = db.session.scalar(
        select(DemoAccess.id).where(DemoAccess.demo_user_id == user.id).limit(1)
    )
    if demo is not None:
        return DataOriginDecision(
            "demo", "demo_access", "explicit", {"demo_access_id": demo}
        )

    heur = _email_heuristic_origin(getattr(user, "email", None))
    if heur:
        return heur

    return DataOriginDecision("unknown", "default", "heuristic", {})


def classify_organization_data_origin(
    *,
    organization_type: str,
    company: Company | None = None,
    institution: Institution | None = None,
    member_user_ids: list[int] | None = None,
) -> DataOriginDecision:
    """Origine org explicite — ne recalcule pas depuis les membres pour promouvoir production."""
    if organization_type == "company" and company is not None:
        override = _override_value("company", int(company.id), "data_origin")
        if override:
            return DataOriginDecision(
                override, "explicit_admin", "explicit", {"override": override}
            )
        demo = db.session.scalar(
            select(DemoAccess.id)
            .where(DemoAccess.demo_company_id == company.id)
            .limit(1)
        )
        if demo is not None:
            return DataOriginDecision(
                "demo", "demo_access", "explicit", {"demo_access_id": demo}
            )
        owner = db.session.get(User, company.user_id) if company.user_id else None
        if owner:
            owner_dec = classify_user_data_origin(owner)
            if owner_dec.data_origin != "unknown":
                return DataOriginDecision(
                    owner_dec.data_origin,
                    f"owner_{owner_dec.source}",
                    owner_dec.confidence,
                    {"via": "owner", **owner_dec.evidence},
                )
            heur = _email_heuristic_origin(getattr(company, "contact_email", None))
            if heur:
                return heur
        return DataOriginDecision("unknown", "default", "heuristic", {})

    if organization_type == "institution" and institution is not None:
        override = _override_value("institution", int(institution.id), "data_origin")
        if override:
            return DataOriginDecision(
                override, "explicit_admin", "explicit", {"override": override}
            )
        # Si TOUS les membres connus sont non-production → org non-production ;
        # sinon unknown (jamais production auto).
        if member_user_ids:
            origins = []
            for uid in member_user_ids:
                u = db.session.get(User, uid)
                if u:
                    origins.append(classify_user_data_origin(u).data_origin)
            non_prod = {"demo", "test", "internal", "synthetic"}
            if origins and all(o in non_prod for o in origins):
                return DataOriginDecision(
                    origins[0],
                    "all_members_non_production",
                    "derived",
                    {"origins": origins},
                )
        return DataOriginDecision("unknown", "default", "heuristic", {})

    return DataOriginDecision("unknown", "default", "heuristic", {})


def apply_user_classification(user: User, decision: DataOriginDecision) -> None:
    """Persiste la classification sauf si override explicite déjà posé avec confidence explicit."""
    if (
        user.data_origin_source == "explicit_admin"
        and user.data_origin_confidence == "explicit"
    ):
        return
    user.data_origin = decision.data_origin
    user.data_origin_source = decision.source
    user.data_origin_confidence = decision.confidence
    user.classified_at = datetime.now(UTC)
    user.classification_evidence_json = decision.evidence
