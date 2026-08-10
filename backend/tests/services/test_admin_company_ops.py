"""Tests gouvernance ops Company (approval / dispatch) + manage-context COMPANY."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from sqlalchemy import select

from ext import db
from models.company import Company
from models.driver import Driver
from models.enums import UserRole
from models.user import User
from security.audit_log import AuditLog
from services.admin_account_manage_context import build_account_manage_context
from services.admin_company_ops import (
    AdminCompanyOpsError,
    set_company_approval,
    set_company_dispatch,
)
from services.control_plane.seed import seed_control_plane_catalogs


def _user(**kwargs) -> User:
    suffix = uuid.uuid4().hex[:10]
    u = User()
    u.public_id = str(uuid.uuid4())
    base = kwargs.get("username", "u")
    u.username = f"{base}_{suffix}"
    email = kwargs.get("email", f"{base}@x.ch")
    local, _, domain = email.partition("@")
    u.email = f"{local}_{suffix}@{domain or 'x.ch'}"
    u.role = kwargs.get("role", UserRole.COMPANY)
    u.set_password("TestPass123!")
    u.account_status = "active"
    db.session.add(u)
    db.session.flush()
    return u


def _company(owner: User, **kwargs) -> Company:
    c = Company()
    c.name = kwargs.get("name", f"Co {owner.id}")
    c.user_id = owner.id
    c.is_approved = kwargs.get("is_approved", True)
    c.dispatch_enabled = kwargs.get("dispatch_enabled", True)
    c.platform_suspended = kwargs.get("platform_suspended", False)
    c.contact_email = owner.email
    c.platform_billing_access_state = kwargs.get(
        "platform_billing_access_state", "active"
    )
    db.session.add(c)
    db.session.flush()
    return c


@pytest.fixture()
def seeded(db_session):
    seed_control_plane_catalogs(commit=False)
    db.session.flush()
    return True


def test_manage_context_company_profile_and_restriction(db_session, seeded):
    from datetime import UTC, datetime

    from services.platform_billing.capabilities import set_billing_access_state

    owner = _user(role=UserRole.COMPANY, username="own", email="own@x.ch")
    company = _company(owner)
    set_billing_access_state(
        company.id,
        "partial",
        source="admin_manual",
        reason_code="test_partial",
        force=True,
    )
    company.platform_billing_state_since = datetime.now(UTC)
    db.session.flush()
    d_active = Driver()
    d_active.user_id = _user(role=UserRole.DRIVER, username="da").id
    d_active.company_id = company.id
    d_active.is_active = True
    d_inactive = Driver()
    d_inactive.user_id = _user(role=UserRole.DRIVER, username="di").id
    d_inactive.company_id = company.id
    d_inactive.is_active = False
    db.session.add_all([d_active, d_inactive])
    db.session.flush()

    admin = _user(role=UserRole.ADMIN, username="adm", email="adm@x.ch")
    with patch(
        "services.admin_account_manage_context.user_has_admin_capability",
        return_value=True,
    ):
        ctx = build_account_manage_context(owner.id, actor_admin_id=admin.id)

    assert ctx is not None
    assert ctx["company_profile"]["company_id"] == company.id
    assert ctx["company_profile"]["active_drivers_count"] == 1
    assert ctx["company_profile"]["total_drivers_count"] == 2
    assert ctx["company_profile"]["inactive_drivers_count"] == 1
    assert "max_drivers" not in ctx["company_profile"]
    assert "quota" not in str(ctx["company_profile"]).lower()
    assert ctx["commercial_restriction"]["state"] == "partial"
    assert ctx["allowed_actions"]["manage_commercial_restriction"] is True
    assert ctx["allowed_actions"]["manage_operational_flags"] is True
    assert ctx["detected_services"]["decision_mode"] == "shadow"
    assert company.platform_suspended is False


def test_billing_shell_not_exposed_as_tenant(db_session, seeded):
    """Shell facturation (non TRANSPORT_TENANT) ne doit pas remplir company_profile."""
    owner = _user(role=UserRole.COMPANY, username="clinic", email="clinic@x.ch")
    c = _company(owner, is_approved=False, dispatch_enabled=False)
    admin = _user(role=UserRole.ADMIN, username="adm2", email="adm2@x.ch")

    from services.control_plane.classification import (
        CompanyProjectionDecision,
        CompanyProjectionKind,
    )

    with (
        patch(
            "services.admin_account_manage_context.classify_company_for_control_plane",
            return_value=CompanyProjectionDecision(
                kind=CompanyProjectionKind.BILLING_SHELL,
                reason="test_shell",
                evidence={},
            ),
        ),
        patch(
            "services.admin_account_manage_context.user_has_admin_capability",
            return_value=True,
        ),
    ):
        ctx = build_account_manage_context(owner.id, actor_admin_id=admin.id)

    assert ctx["company_profile"] is None
    assert ctx["commercial_restriction"] is None
    assert c.id is not None


def test_approval_independent_of_dispatch(db_session, seeded):
    owner = _user(role=UserRole.COMPANY, username="own3", email="own3@x.ch")
    company = _company(owner, is_approved=False, dispatch_enabled=False)
    admin = _user(role=UserRole.ADMIN, username="adm3", email="adm3@x.ch")

    with patch("services.admin_company_ops.get_projector") as proj:
        proj.return_value.ensure_company_organization.return_value = None
        result = set_company_approval(
            company_id=company.id,
            is_approved=True,
            reason="Dossier contractuel validé",
            actor_admin_id=admin.id,
            expected_is_approved=False,
        )
    assert result.status == "updated"
    db.session.refresh(company)
    assert company.is_approved is True
    assert company.dispatch_enabled is False
    assert company.platform_suspended is False

    audit = db.session.scalar(
        select(AuditLog)
        .where(AuditLog.action_type == "admin_company_approved")
        .order_by(AuditLog.id.desc())
    )
    assert audit is not None


def test_dispatch_disable_no_platform_suspend(db_session, seeded):
    owner = _user(role=UserRole.COMPANY, username="own4", email="own4@x.ch")
    company = _company(
        owner,
        is_approved=True,
        dispatch_enabled=True,
        platform_billing_access_state="active",
    )
    admin = _user(role=UserRole.ADMIN, username="adm4", email="adm4@x.ch")

    with patch("services.admin_company_ops.get_projector") as proj:
        proj.return_value.ensure_company_organization.return_value = None
        result = set_company_dispatch(
            company_id=company.id,
            dispatch_enabled=False,
            reason="Arrêt temporaire de l’exploitation",
            actor_admin_id=admin.id,
            expected_dispatch_enabled=True,
        )
    assert result.status == "updated"
    db.session.refresh(company)
    assert company.dispatch_enabled is False
    assert company.platform_suspended is False
    assert company.is_approved is True
    assert company.platform_billing_access_state == "active"


def test_approval_concurrent_conflict(db_session, seeded):
    owner = _user(role=UserRole.COMPANY, username="own5", email="own5@x.ch")
    company = _company(owner, is_approved=True)
    admin = _user(role=UserRole.ADMIN, username="adm5", email="adm5@x.ch")

    with pytest.raises(AdminCompanyOpsError) as exc:
        set_company_approval(
            company_id=company.id,
            is_approved=False,
            reason="Changement concurrent",
            actor_admin_id=admin.id,
            expected_is_approved=False,
        )
    assert exc.value.status_code == 409


def test_dispatch_noop(db_session, seeded):
    owner = _user(role=UserRole.COMPANY, username="own6", email="own6@x.ch")
    company = _company(owner, dispatch_enabled=True)
    admin = _user(role=UserRole.ADMIN, username="adm6", email="adm6@x.ch")

    with patch("services.admin_company_ops.get_projector") as proj:
        result = set_company_dispatch(
            company_id=company.id,
            dispatch_enabled=True,
            reason="Déjà activé aujourd'hui",
            actor_admin_id=admin.id,
            expected_dispatch_enabled=True,
        )
    assert result.status == "unchanged"
    proj.return_value.ensure_company_organization.assert_not_called()
