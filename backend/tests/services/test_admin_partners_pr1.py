"""Tests PR1 Partenaires — normalisation rôle, gel mutations, organisations."""

from __future__ import annotations

import uuid

import pytest

from models import Company, User
from models.enums import UserRole
from services.admin_role_utils import normalized_role_value


def _make_user(*, username: str, email: str, role: UserRole) -> User:
    user = User()
    user.username = username
    user.email = email
    user.role = role
    user.public_id = str(uuid.uuid4())
    user.set_password("Password1!", force_change=False)
    return user


@pytest.mark.parametrize(
    "role,expected",
    [
        (UserRole.COMPANY, "COMPANY"),
        (UserRole.company, "COMPANY"),
        ("COMPANY", "COMPANY"),
        ("company", "COMPANY"),
        (" Company ", "COMPANY"),
        (UserRole.ADMIN, "ADMIN"),
        (None, ""),
    ],
)
def test_normalized_role_value(role, expected):
    assert normalized_role_value(role) == expected


def test_enrich_users_admin_payload_matches_company_role(app, db_session):
    from routes.admin import _enrich_users_admin_payload

    with app.app_context():
        user = _make_user(
            username=f"partner_enrich_{uuid.uuid4().hex[:8]}",
            email=f"partner_enrich_{uuid.uuid4().hex[:8]}@example.com",
            role=UserRole.COMPANY,
        )
        db_session.session.add(user)
        db_session.session.flush()
        company = Company(user_id=user.id, name="Partner Enrich Co")
        db_session.session.add(company)
        db_session.session.flush()

        payload = _enrich_users_admin_payload([user])
        assert len(payload) == 1
        assert payload[0]["company_id"] == company.id
        assert payload[0]["company_name"] == "Partner Enrich Co"


def test_update_user_role_same_role_noop(client, admin_headers, db_session, app):
    with app.app_context():
        target = _make_user(
            username=f"role_noop_{uuid.uuid4().hex[:8]}",
            email=f"role_noop_{uuid.uuid4().hex[:8]}@example.com",
            role=UserRole.CLIENT,
        )
        db_session.session.add(target)
        db_session.session.commit()
        user_id = target.id

    response = client.put(
        f"/api/v1/admin/users/{user_id}/role",
        json={
            "role": "client",
            "expected_current_role": "client",
            "reason": "No-op de vérification API",
        },
        headers=admin_headers,
    )
    assert response.status_code == 200


def test_update_user_role_apply_requires_reason(client, admin_headers, db_session, app):
    """Apply sans reason → 400 (schema durci)."""
    with app.app_context():
        target = _make_user(
            username=f"role_ok_{uuid.uuid4().hex[:8]}",
            email=f"role_ok_{uuid.uuid4().hex[:8]}@example.com",
            role=UserRole.CLIENT,
        )
        db_session.session.add(target)
        db_session.session.commit()
        user_id = target.id

    response = client.put(
        f"/api/v1/admin/users/{user_id}/role",
        json={"role": "client", "expected_current_role": "client"},
        headers=admin_headers,
    )
    assert response.status_code == 400


def test_delete_user_blocked_when_not_testing(client, admin_headers, db_session, app):
    with app.app_context():
        target = _make_user(
            username=f"delete_block_{uuid.uuid4().hex[:8]}",
            email=f"delete_block_{uuid.uuid4().hex[:8]}@example.com",
            role=UserRole.CLIENT,
        )
        db_session.session.add(target)
        db_session.session.commit()
        user_id = target.id

    previous = app.config.get("TESTING")
    app.config["TESTING"] = False
    try:
        response = client.delete(
            f"/api/v1/admin/users/{user_id}",
            headers=admin_headers,
        )
        assert response.status_code == 409
        assert response.get_json().get("error") == "physical_user_deletion_requires_review"
    finally:
        app.config["TESTING"] = previous


def test_partners_organizations_list(client, admin_headers, db_session, app):
    with app.app_context():
        user = _make_user(
            username=f"org_list_{uuid.uuid4().hex[:8]}",
            email=f"org_list_{uuid.uuid4().hex[:8]}@example.com",
            role=UserRole.COMPANY,
        )
        db_session.session.add(user)
        db_session.session.flush()
        db_session.session.add(Company(user_id=user.id, name="Org List Co"))
        db_session.session.commit()

    response = client.get(
        "/api/v1/admin/partners/organizations?include_synthetic=true&per_page=50",
        headers=admin_headers,
    )
    assert response.status_code == 200
    body = response.get_json()
    assert "items" in body
    assert "summary" in body
    assert body["summary_scope"]["demonstrations_include_all"] is True


def test_account_integrity_orphan_company(client, admin_headers, db_session, app):
    with app.app_context():
        orphan = _make_user(
            username=f"orphan_co_{uuid.uuid4().hex[:8]}",
            email=f"orphan_co_{uuid.uuid4().hex[:8]}@example.com",
            role=UserRole.COMPANY,
        )
        db_session.session.add(orphan)
        db_session.session.commit()
        user_id = orphan.id

    response = client.get(
        f"/api/v1/admin/partners/accounts/{user_id}/integrity",
        headers=admin_headers,
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["configuration_status"] == "incomplete"
    codes = {c["code"]: c["status"] for c in body["checks"]}
    assert codes.get("COMPANY_PROFILE_LINKED") == "failed"


def test_company_user_id_shared_shell_not_unique(app, db_session):
    """Plusieurs Company peuvent partager un user_id (clinic shells).

    Une seule peut être tenant ; la shell n'est pas projetée ; owner COMPANY
    + clinic sans chauffeur → ambiguous (fail-closed).
    """
    from models.billing_party import BillingParty
    from models.clinic_billing_party_mapping import ClinicBillingPartyMapping
    from models.driver import Driver
    from models.enums import BillingPartyType
    from services.control_plane.classification import (
        CompanyProjectionKind,
        classify_company_for_control_plane,
    )
    from services.control_plane.projector import ControlPlaneProjector
    from services.control_plane.seed import seed_control_plane_catalogs

    with app.app_context():
        seed_control_plane_catalogs(commit=False)
        owner = _make_user(
            username=f"share_owner_{uuid.uuid4().hex[:8]}",
            email=f"share_owner_{uuid.uuid4().hex[:8]}@example.com",
            role=UserRole.COMPANY,
        )
        db_session.session.add(owner)
        db_session.session.flush()

        tenant = Company(user_id=owner.id, name="Tenant Transport")
        shell = Company(user_id=owner.id, name="Clinic Shell")
        db_session.session.add_all([tenant, shell])
        db_session.session.flush()

        driver_user = _make_user(
            username=f"drv_share_{uuid.uuid4().hex[:8]}",
            email=f"drv_share_{uuid.uuid4().hex[:8]}@example.com",
            role=UserRole.DRIVER,
        )
        db_session.session.add(driver_user)
        db_session.session.flush()
        d = Driver()
        d.user_id = driver_user.id
        d.company_id = tenant.id
        d.is_active = True
        db_session.session.add(d)
        db_session.session.flush()

        bp = BillingParty()
        bp.company_id = tenant.id
        bp.type = BillingPartyType.CLINIC
        bp.display_name = "Clinic BP"
        bp.external_ref = f"clinic_company:{shell.id}"
        db_session.session.add(bp)
        db_session.session.flush()
        mapping = ClinicBillingPartyMapping()
        mapping.company_id = tenant.id
        mapping.clinic_company_id = shell.id
        mapping.billing_party_id = bp.id
        db_session.session.add(mapping)
        db_session.session.flush()

        assert classify_company_for_control_plane(tenant).kind == (
            CompanyProjectionKind.TRANSPORT_TENANT
        )
        # Owner COMPANY + clinic + 0 driver → ambiguous (pas shell silencieuse)
        assert classify_company_for_control_plane(shell).kind == (
            CompanyProjectionKind.AMBIGUOUS
        )

        proj = ControlPlaneProjector()
        assert proj.ensure_company_organization(tenant) is not None
        assert proj.ensure_company_organization(shell) is None
        db_session.session.commit()
