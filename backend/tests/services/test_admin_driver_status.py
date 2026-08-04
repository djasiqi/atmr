"""Tests soft-disable chauffeur admin + revoke sessions + enrich liste."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from ext import db
from models.company import Company
from models.control_plane import OrganizationMembership, RoleTemplate
from models.driver import Driver
from models.enums import UserRole
from models.user import User
from services.admin_driver_status import (
    AdminDriverStatusError,
    set_driver_status,
)
from services.control_plane.projector import get_projector
from services.control_plane.seed import seed_control_plane_catalogs


def _user(**kwargs) -> User:
    suffix = uuid.uuid4().hex[:10]
    u = User()
    u.public_id = str(uuid.uuid4())
    base = kwargs.get("username", "drv")
    u.username = f"{base}_{suffix}"
    email = kwargs.get("email", f"{base}@x.ch")
    local, _, domain = email.partition("@")
    u.email = f"{local}_{suffix}@{domain or 'x.ch'}"
    u.role = kwargs.get("role", UserRole.DRIVER)
    u.set_password("TestPass123!")
    u.account_status = "active"
    db.session.add(u)
    db.session.flush()
    return u


def _company(owner: User) -> Company:
    c = Company()
    c.name = f"Co {owner.id}"
    c.user_id = owner.id
    c.is_approved = True
    c.dispatch_enabled = True
    c.contact_email = owner.email
    db.session.add(c)
    db.session.flush()
    return c


@pytest.fixture()
def seeded(db_session):
    seed_control_plane_catalogs(commit=False)
    db.session.flush()
    return True


def _make_driver_tenant(seeded):
    owner = _user(role=UserRole.COMPANY, username="own", email="own@x.ch")
    company = _company(owner)
    u = _user(role=UserRole.DRIVER, username="drv", email="drv@x.ch")
    d = Driver()
    d.user_id = u.id
    d.company_id = company.id
    d.is_active = True
    d.is_available = True
    db.session.add(d)
    db.session.flush()
    get_projector().sync_driver(d)
    db.session.flush()
    admin = _user(role=UserRole.ADMIN, username="adm", email="adm@x.ch")
    return u, d, company, admin


def test_disable_suspends_cp_membership(db_session, seeded):
    u, d, _company, admin = _make_driver_tenant(seeded)
    with patch(
        "security.mobile_device_session_service.revoke_user_security_sessions",
        return_value=1,
    ):
        result = set_driver_status(
            user_id=u.id,
            is_active=False,
            reason="Fin de mission support",
            actor_admin_id=admin.id,
            expected_is_active=True,
        )
    assert result.status == "updated"
    assert result.is_active is False
    assert result.is_available is False
    db.session.refresh(d)
    assert d.is_active is False
    assert d.is_available is False

    role = db.session.scalar(
        select(RoleTemplate).where(
            RoleTemplate.organization_type == "company",
            RoleTemplate.role_key == "company_driver",
        )
    )
    m = db.session.scalar(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == u.id,
            OrganizationMembership.role_template_id == role.id,
        )
    )
    assert m is not None
    assert m.membership_status == "suspended"
    assert m.suspended_at is not None


def test_reactivate_keeps_available_false(db_session, seeded):
    u, d, _c, admin = _make_driver_tenant(seeded)
    with patch(
        "security.mobile_device_session_service.revoke_user_security_sessions",
        return_value=1,
    ):
        set_driver_status(
            user_id=u.id,
            is_active=False,
            reason="Pause temporaire chauffeur",
            actor_admin_id=admin.id,
        )
    result = set_driver_status(
        user_id=u.id,
        is_active=True,
        reason="Retour du collaborateur",
        actor_admin_id=admin.id,
        expected_is_active=False,
    )
    assert result.is_active is True
    assert result.is_available is False
    db.session.refresh(d)
    assert d.is_available is False

    role = db.session.scalar(
        select(RoleTemplate).where(
            RoleTemplate.organization_type == "company",
            RoleTemplate.role_key == "company_driver",
        )
    )
    m = db.session.scalar(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == u.id,
            OrganizationMembership.role_template_id == role.id,
        )
    )
    assert m.membership_status == "active"
    assert m.suspended_at is None


def test_noop_same_status(db_session, seeded):
    u, d, _c, admin = _make_driver_tenant(seeded)
    with patch(
        "security.mobile_device_session_service.revoke_user_security_sessions"
    ) as revoke_mock:
        result = set_driver_status(
            user_id=u.id,
            is_active=True,
            reason="No-op volontaire test",
            actor_admin_id=admin.id,
            expected_is_active=True,
        )
    assert result.status == "unchanged"
    assert result.sessions_revoked == 0
    revoke_mock.assert_not_called()


def test_expected_mismatch_409(db_session, seeded):
    u, _d, _c, admin = _make_driver_tenant(seeded)
    with pytest.raises(AdminDriverStatusError) as exc:
        set_driver_status(
            user_id=u.id,
            is_active=False,
            reason="Concurrence attendue",
            actor_admin_id=admin.id,
            expected_is_active=False,
        )
    assert exc.value.status_code == 409
    assert exc.value.error == "driver_status_changed"


def test_revoke_fail_closed_rolls_back(db_session, seeded):
    u, d, _c, admin = _make_driver_tenant(seeded)
    driver_id = d.id
    with patch(
        "security.mobile_device_session_service.revoke_user_security_sessions",
        side_effect=RuntimeError("boom"),
    ):
        with pytest.raises(AdminDriverStatusError) as exc:
            set_driver_status(
                user_id=u.id,
                is_active=False,
                reason="Échec revoke attendu",
                actor_admin_id=admin.id,
            )
    assert exc.value.status_code == 503
    refreshed = db.session.get(Driver, driver_id)
    assert refreshed is not None
    assert refreshed.is_active is True


def test_enrich_list_driver_has_company_no_billing(app, db_session, seeded):
    from routes.admin import _enrich_users_admin_payload

    u, d, company, _admin = _make_driver_tenant(seeded)
    db.session.commit()
    with app.app_context():
        payload = _enrich_users_admin_payload([u])
    assert len(payload) == 1
    row = payload[0]
    assert row["company_id"] == company.id
    assert row["company_name"] == company.name
    assert row["driver_id"] == d.id
    assert row["driver_is_active"] is True
    assert "platform_billing_access_state" not in row


def test_manage_context_driver_actions(app, db_session, seeded):
    from services.admin_account_manage_context import build_account_manage_context

    u, _d, company, admin = _make_driver_tenant(seeded)
    with patch(
        "services.admin_account_manage_context.user_has_admin_capability",
        return_value=True,
    ):
        ctx = build_account_manage_context(u.id, actor_admin_id=admin.id)
    assert ctx is not None
    assert ctx["driver_profile"]["company_name"] == company.name
    assert ctx["allowed_actions"]["change_driver_status"] is True
    assert ctx["allowed_actions"]["manage_billing_access"] is False
    assert ctx["allowed_actions"]["revoke_sessions"] is True
    assert ctx["commercial_access"] is None
