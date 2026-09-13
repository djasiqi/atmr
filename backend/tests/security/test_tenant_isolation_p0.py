"""Tests Lot 0 P0 — isolation tenant (SEC-04 / SEC-05)."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest
from flask_jwt_extended import decode_token


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def company_a(db):
    from models import Company, User
    from models.enums import UserRole

    uname = _unique("company_a_p0")
    user = User(
        username=uname,
        email=f"{uname}@example.com",
        role=UserRole.COMPANY,
    )
    user.set_password("SecurePass1!")
    db.session.add(user)
    db.session.flush()
    company = Company(name=f"Company A {uname}", user_id=user.id)
    db.session.add(company)
    db.session.commit()
    db.session.refresh(user)
    return company, user


@pytest.fixture
def company_b(db):
    from models import Company, User
    from models.enums import UserRole

    uname = _unique("company_b_p0")
    user = User(
        username=uname,
        email=f"{uname}@example.com",
        role=UserRole.COMPANY,
    )
    user.set_password("SecurePass1!")
    db.session.add(user)
    db.session.flush()
    company = Company(name=f"Company B {uname}", user_id=user.id)
    db.session.add(company)
    db.session.commit()
    db.session.refresh(user)
    return company, user


@pytest.fixture
def admin_user(db):
    from models import User
    from models.enums import UserRole

    uname = _unique("admin_p0")
    user = User(
        username=uname,
        email=f"{uname}@example.com",
        role=UserRole.ADMIN,
    )
    user.set_password("SecurePass1!")
    db.session.add(user)
    db.session.commit()
    return user


def _business_session_headers(app, user):
    """Session métier déjà authentifiée (équivalent post-/totp/challenge).

    Les tests tenant vérifient l'autorisation, pas le parcours login/MFA.
    """
    from routes.auth import _resolve_company_id, issue_business_access_token

    with app.app_context():
        token = issue_business_access_token(user)
        claims = decode_token(token)
    assert claims.get("sub") == str(user.public_id)
    assert claims.get("role") == (
        user.role.value if hasattr(user.role, "value") else str(user.role)
    )
    assert claims.get("company_id") == _resolve_company_id(user)
    assert not claims.get("purpose")
    return {"Authorization": f"Bearer {token}"}


class TestInvoicesListAuth:
    def test_anonymous_gets_401(self, client, company_a):
        company, _ = company_a
        resp = client.get(f"/api/v1/invoices/companies/{company.id}/invoices")
        assert resp.status_code in (401, 422)

    def test_company_a_cannot_access_company_b(self, client, app, company_a, company_b):
        _, user_a = company_a
        company_b_obj, _ = company_b
        headers = _business_session_headers(app, user_a)
        resp = client.get(
            f"/api/v1/invoices/companies/{company_b_obj.id}/invoices",
            headers=headers,
        )
        assert resp.status_code == 403

    def test_company_a_can_access_own(self, client, app, company_a):
        company, user_a = company_a
        headers = _business_session_headers(app, user_a)
        resp = client.get(
            f"/api/v1/invoices/companies/{company.id}/invoices",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_admin_cross_tenant_allowed_and_audited(
        self, client, app, company_a, admin_user
    ):
        company, _ = company_a
        headers = _business_session_headers(app, admin_user)
        with patch("shared.tenant_guard.AuditLogger.log_action") as mock_audit:
            resp = client.get(
                f"/api/v1/invoices/companies/{company.id}/invoices",
                headers=headers,
            )
            assert resp.status_code == 200
            assert mock_audit.called
            kwargs = mock_audit.call_args.kwargs
            assert kwargs.get("action_type") == "admin_cross_tenant_access"


class TestInvoicesDebugIdor:
    def test_company_b_blocked(self, client, app, company_a, company_b):
        company_a_obj, _ = company_a
        _, user_b = company_b
        headers = _business_session_headers(app, user_b)
        resp = client.get(
            f"/api/v1/invoices/companies/{company_a_obj.id}/invoices/debug",
            headers=headers,
        )
        assert resp.status_code == 403


class TestExportPaymentsIdor:
    def test_company_b_blocked(self, client, app, company_a, company_b):
        company_a_obj, _ = company_a
        _, user_b = company_b
        headers = _business_session_headers(app, user_b)
        resp = client.get(
            f"/api/v1/invoices/companies/{company_a_obj.id}/exports/payments.csv"
            f"?year=2026&month=1",
            headers=headers,
        )
        assert resp.status_code == 403


class TestBusinessSessionFixture:
    def test_helper_issues_access_token_accepted_by_jwt_required(
        self, client, app, company_a
    ):
        _, user_a = company_a
        headers = _business_session_headers(app, user_a)
        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code == 200
        body = me.get_json() or {}
        assert str(body.get("public_id")) == str(user_a.public_id)
