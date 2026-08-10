"""Tests transitions de rôle admin sécurisées + reset MDP."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from ext import db
from models.company import Company
from models.driver import Driver
from models.enums import UserRole
from models.user import User
from services.admin_account_role_transition import (
    AdminAccountRoleTransitionService,
)
from services.control_plane.seed import seed_control_plane_catalogs


def _user(**kwargs) -> User:
    suffix = uuid.uuid4().hex[:10]
    u = User()
    u.public_id = str(uuid.uuid4())
    u.username = kwargs.get("username", f"u_{suffix}")
    if not kwargs.get("username"):
        u.username = f"u_{suffix}"
    else:
        # Préfixe unique pour éviter les collisions après commit apply
        u.username = f"{kwargs['username']}_{suffix}"
    email = kwargs.get("email")
    if email and "@" in email:
        local, _, domain = email.partition("@")
        u.email = f"{local}_{suffix}@{domain}"
    else:
        u.email = f"{u.username}@prod.example.ch"
    u.role = kwargs.get("role", UserRole.CLIENT)
    u.set_password("TestPass123!")
    u.account_status = "active"
    db.session.add(u)
    db.session.flush()
    return u


def _company(owner: User, **kwargs) -> Company:
    c = Company()
    c.name = kwargs.get("name", f"Co {owner.id}")
    c.user_id = owner.id
    c.is_approved = True
    c.dispatch_enabled = True
    c.contact_email = owner.email
    db.session.add(c)
    db.session.flush()
    return c


@pytest.fixture
def seeded(db_session):
    seed_control_plane_catalogs(commit=False)
    db.session.flush()
    return True


def test_noop_same_role_same_context(db_session, seeded):
    u = _user(role=UserRole.CLIENT)
    svc = AdminAccountRoleTransitionService()
    preview = svc.preview(user_id=u.id, target_role="client")
    assert preview.allowed is True
    assert any("Aucun changement" in c for c in preview.changes)


def test_client_to_driver_requires_company(db_session, seeded):
    u = _user(role=UserRole.CLIENT)
    svc = AdminAccountRoleTransitionService()
    preview = svc.preview(user_id=u.id, target_role="driver")
    assert preview.allowed is False
    assert preview.blockers[0]["code"] == "company_id_required"


def test_client_to_driver_rejects_non_tenant_shell(db_session, seeded):
    shell_owner = _user(role=UserRole.CLIENT, username="shellown", email="s@x.ch")
    shell = _company(shell_owner, name="Shellish")
    from services.control_plane.classification import (
        CompanyProjectionKind,
        classify_company_for_control_plane,
    )

    kind = classify_company_for_control_plane(shell).kind
    if kind == CompanyProjectionKind.TRANSPORT_TENANT:
        pytest.skip("classification a résolu un tenant — skip")

    u = _user(role=UserRole.CLIENT, username="tocdrv", email="toc@x.ch")
    svc = AdminAccountRoleTransitionService()
    preview = svc.preview(user_id=u.id, target_role="driver", company_id=shell.id)
    assert preview.allowed is False
    assert preview.blockers[0]["code"] == "company_not_transport_tenant"


def test_client_to_driver_ok(db_session, seeded):
    owner = _user(role=UserRole.COMPANY, username="own1", email="own1@x.ch")
    company = _company(owner)
    # make transport tenant via driver
    du = _user(role=UserRole.DRIVER, username="exd", email="exd@x.ch")
    d = Driver()
    d.user_id = du.id
    d.company_id = company.id
    d.is_active = True
    db.session.add(d)
    db.session.flush()

    u = _user(role=UserRole.CLIENT, username="newdrv", email="newdrv@x.ch")
    admin = _user(role=UserRole.ADMIN, username="adm", email="adm@x.ch")
    svc = AdminAccountRoleTransitionService()
    with (
        patch(
            "services.admin_account_role_transition.user_has_admin_capability",
            return_value=True,
        ),
        patch("security.mobile_device_session_service.revoke_user_security_sessions"),
    ):
        result = svc.apply(
            user_id=u.id,
            target_role="driver",
            expected_current_role="client",
            reason="Affectation nouveau chauffeur",
            actor_admin_id=admin.id,
            company_id=company.id,
        )
    assert result.noop is False
    db.session.refresh(u)
    assert str(u.role).upper().endswith("DRIVER") or u.role == UserRole.DRIVER
    drv = Driver.query.filter_by(user_id=u.id).first()
    assert drv is not None
    assert drv.company_id == company.id
    assert drv.is_active is True


def test_driver_to_client_soft_disables(db_session, seeded):
    owner = _user(role=UserRole.COMPANY, username="own2", email="own2@x.ch")
    company = _company(owner)
    u = _user(role=UserRole.DRIVER, username="drv2", email="drv2@x.ch")
    d = Driver()
    d.user_id = u.id
    d.company_id = company.id
    d.is_active = True
    d.is_available = True
    db.session.add(d)
    # need another driver so company stays tenant
    other = _user(role=UserRole.DRIVER, username="drv3", email="drv3@x.ch")
    d2 = Driver()
    d2.user_id = other.id
    d2.company_id = company.id
    d2.is_active = True
    db.session.add(d2)
    db.session.flush()

    admin = _user(role=UserRole.ADMIN, username="adm2", email="adm2@x.ch")
    svc = AdminAccountRoleTransitionService()
    with patch("security.mobile_device_session_service.revoke_user_security_sessions"):
        svc.apply(
            user_id=u.id,
            target_role="client",
            expected_current_role="driver",
            expected_company_id=company.id,
            reason="Fin de mission chauffeur",
            actor_admin_id=admin.id,
        )
    db.session.refresh(d)
    assert Driver.query.get(d.id) is not None
    assert d.is_active is False
    assert d.is_available is False


def test_company_owner_cannot_leave_without_cp_pr3(db_session, seeded):
    owner = _user(role=UserRole.COMPANY, username="own3", email="own3@x.ch")
    company = _company(owner)
    du = _user(role=UserRole.DRIVER, username="dx", email="dx@x.ch")
    d = Driver()
    d.user_id = du.id
    d.company_id = company.id
    d.is_active = True
    db.session.add(d)
    db.session.flush()

    svc = AdminAccountRoleTransitionService()
    preview = svc.preview(user_id=owner.id, target_role="client")
    assert preview.allowed is False
    assert preview.blockers[0]["code"] == "company_ownership_transition_required"


def test_to_company_without_tenant_409(db_session, seeded):
    u = _user(role=UserRole.CLIENT, username="nocomp", email="nocomp@x.ch")
    svc = AdminAccountRoleTransitionService()
    preview = svc.preview(user_id=u.id, target_role="company")
    assert preview.allowed is False
    assert preview.blockers[0]["code"] == "company_owner_assignment_required"


def test_reset_password_requires_reason(client, admin_headers, db_session, app):
    with app.app_context():
        u = _user(role=UserRole.CLIENT, username="rp1", email="rp1@x.ch")
        db.session.commit()
        uid = u.id
    response = client.post(
        f"/api/v1/admin/users/{uid}/reset-password",
        json={},
        headers=admin_headers,
    )
    assert response.status_code == 400


def test_delete_still_blocked(client, admin_headers, db_session, app):
    with app.app_context():
        u = _user(role=UserRole.CLIENT, username="del1", email="del1@x.ch")
        db.session.commit()
        uid = u.id
    previous = app.config.get("TESTING")
    app.config["TESTING"] = False
    try:
        response = client.delete(
            f"/api/v1/admin/users/{uid}",
            headers=admin_headers,
        )
        assert response.status_code == 409
    finally:
        app.config["TESTING"] = previous
