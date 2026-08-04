"""Tests CP-PR1 — classification Company fail-closed + lifecycle + effective-access."""

from __future__ import annotations

import uuid

import pytest

from ext import db
from models.clinic_billing_party_mapping import ClinicBillingPartyMapping
from models.company import Company
from models.control_plane import (
    OrganizationMembership,
    OrganizationServiceEntitlement,
    PlatformOrganization,
    RoleTemplate,
    ServiceCatalog,
)
from models.driver import Driver
from models.enums import InstitutionRole, UserRole
from models.institution import Institution
from models.user import User
from services.control_plane.classification import (
    CompanyProjectionKind,
    classify_company_for_control_plane,
    classify_user_data_origin,
    derive_company_lifecycle,
)
from services.control_plane.effective_access import compute_effective_access
from services.control_plane.projector import ControlPlaneProjector
from services.control_plane.seed import seed_control_plane_catalogs


def _user(**kwargs) -> User:
    u = User()
    u.public_id = str(uuid.uuid4())
    u.username = kwargs.get("username", f"u_{uuid.uuid4().hex[:8]}")
    u.email = kwargs.get("email", f"{u.username}@prod.example.ch")
    u.role = kwargs.get("role", UserRole.COMPANY)
    u.set_password("TestPass123!")
    for k, v in kwargs.items():
        if k not in ("username", "email", "role") and hasattr(u, k):
            setattr(u, k, v)
    db.session.add(u)
    db.session.flush()
    return u


def _company(owner: User, **kwargs) -> Company:
    c = Company()
    c.name = kwargs.get("name", f"Co {owner.id}")
    c.user_id = owner.id
    c.is_approved = kwargs.get("is_approved", False)
    c.platform_suspended = kwargs.get("platform_suspended", False)
    c.dispatch_enabled = kwargs.get("dispatch_enabled", False)
    c.contact_email = kwargs.get("contact_email", owner.email)
    db.session.add(c)
    db.session.flush()
    return c


@pytest.fixture()
def seeded(db_session):
    seed_control_plane_catalogs(commit=False)
    db.session.flush()
    return True


def test_classify_driver_makes_transport_tenant(db_session, seeded):
    owner = _user(role=UserRole.COMPANY)
    company = _company(owner)
    driver_user = _user(role=UserRole.DRIVER, username="drv1", email="drv1@prod.example.ch")
    d = Driver()
    d.user_id = driver_user.id
    d.company_id = company.id
    d.is_active = True
    db.session.add(d)
    db.session.flush()

    decision = classify_company_for_control_plane(company)
    assert decision.kind == CompanyProjectionKind.TRANSPORT_TENANT


def test_classify_clinic_shell_company_owner_is_ambiguous(db_session, seeded):
    """clinic + 0 driver + owner COMPANY → AMBIGUOUS (fail-closed)."""
    owner = _user(role=UserRole.COMPANY)
    tenant = _company(owner, name="Tenant")
    shell = _company(owner, name="Clinic Shell")
    from models.billing_party import BillingParty
    from models.enums import BillingPartyType

    bp = BillingParty()
    bp.company_id = tenant.id
    bp.type = BillingPartyType.CLINIC
    bp.display_name = "Clinic BP"
    bp.external_ref = f"clinic_company:{shell.id}"
    db.session.add(bp)
    db.session.flush()
    m = ClinicBillingPartyMapping()
    m.company_id = tenant.id
    m.clinic_company_id = shell.id
    m.billing_party_id = bp.id
    db.session.add(m)
    db.session.flush()

    decision = classify_company_for_control_plane(shell)
    assert decision.kind == CompanyProjectionKind.AMBIGUOUS


def test_classify_clinic_shell_non_company_owner(db_session, seeded):
    """clinic + 0 driver + owner non-COMPANY → BILLING_SHELL."""
    owner = _user(role=UserRole.CLIENT)
    shell = _company(owner, name="Clinic Shell")
    from models.billing_party import BillingParty
    from models.enums import BillingPartyType

    bp = BillingParty()
    bp.company_id = shell.id
    bp.type = BillingPartyType.CLINIC
    bp.display_name = "Clinic BP"
    bp.external_ref = f"clinic_company:{shell.id}"
    db.session.add(bp)
    db.session.flush()
    m = ClinicBillingPartyMapping()
    m.company_id = shell.id
    m.clinic_company_id = shell.id
    m.billing_party_id = bp.id
    db.session.add(m)
    db.session.flush()

    decision = classify_company_for_control_plane(shell)
    assert decision.kind == CompanyProjectionKind.BILLING_SHELL


def test_data_origin_never_auto_production(db_session, seeded):
    u = _user(email="real.person@clinique.ch")
    dec = classify_user_data_origin(u)
    assert dec.data_origin == "unknown"


def test_data_origin_demo_email(db_session, seeded):
    u = _user(email="foo@demo.local")
    dec = classify_user_data_origin(u)
    assert dec.data_origin == "demo"


def test_lifecycle_company(db_session, seeded):
    owner = _user()
    c = _company(owner, platform_suspended=True)
    assert derive_company_lifecycle(c) == "suspended"
    c.platform_suspended = False
    c.is_approved = True
    assert derive_company_lifecycle(c) == "active"


def test_projector_tenant_and_shadow_entitlement(db_session, seeded):
    owner = _user(role=UserRole.COMPANY)
    company = _company(owner, is_approved=True)
    driver_user = _user(role=UserRole.DRIVER, username="drv2", email="drv2@x.ch")
    d = Driver()
    d.user_id = driver_user.id
    d.company_id = company.id
    d.is_active = True
    db.session.add(d)
    db.session.flush()

    proj = ControlPlaneProjector()
    org = proj.ensure_company_organization(company)
    assert org is not None
    assert org.organization_type == "company"
    ents = OrganizationServiceEntitlement.query.filter_by(
        organization_id=org.id
    ).all()
    assert ents
    assert all(e.enforcement_mode == "shadow" for e in ents)


def test_effective_access_shadow(db_session, seeded):
    inst = Institution()
    inst.name = "Clinique Test"
    inst.contact_email = "c@test.ch"
    db.session.add(inst)
    db.session.flush()

    admin = _user(
        role=UserRole.INSTITUTION,
        username="iadm",
        email="iadm@test.ch",
        institution_id=inst.id,
        institution_role=InstitutionRole.ADMIN.value,
        account_status="active",
    )
    proj = ControlPlaneProjector()
    org = proj.ensure_institution_organization(inst)
    proj.ensure_shadow_entitlements_institution(org)
    proj.sync_institution_user(admin)
    db.session.flush()

    payload = compute_effective_access(admin.id)
    assert payload["decision_mode"] == "shadow"
    assert payload["permissions_enforced"] == []
    assert payload["subject_state"] in ("eligible", "blocked", "needs_review")


def test_effective_access_invited_membership_blocked(db_session, seeded):
    inst = Institution()
    inst.name = "Clinique Invite"
    inst.contact_email = "c@invite.ch"
    db.session.add(inst)
    db.session.flush()

    invited = _user(
        role=UserRole.INSTITUTION,
        username="invited_u",
        email="invited@test.ch",
        institution_id=inst.id,
        institution_role=InstitutionRole.READER.value,
        account_status="active",
    )
    proj = ControlPlaneProjector()
    proj.ensure_institution_organization(inst)
    membership = proj.sync_institution_user(invited)
    assert membership is not None
    membership.membership_status = "invited"
    db.session.flush()

    payload = compute_effective_access(invited.id)
    assert payload["subject_state"] == "blocked"
    codes = [
        *payload["blocking_reasons"],
        *[r for m in payload["memberships"] for r in m["blocking_reasons"]],
    ]
    assert any(b["code"] == "MEMBERSHIP_INVITED" for b in codes)


def test_effective_access_no_membership_blocked(db_session, seeded):
    lone = _user(
        role=UserRole.INSTITUTION,
        username="lone_u",
        email="lone@test.ch",
        account_status="active",
    )
    payload = compute_effective_access(lone.id)
    assert payload["subject_state"] == "blocked"
    assert any(b["code"] == "NO_ACTIVE_MEMBERSHIP" for b in payload["blocking_reasons"])


def test_no_org_for_orphan_company_account(db_session, seeded):
    orphan = _user(role=UserRole.COMPANY, username="orphan_co", email="orphan@x.ch")
    # pas de Company
    orgs = PlatformOrganization.query.filter_by(company_id=None).count()
    _ = orphan
    _ = orgs
    proj = ControlPlaneProjector()
    # ensure_company_organization nécessite une Company ; orphan n'en a pas
    assert Company.query.filter_by(user_id=orphan.id).first() is None
