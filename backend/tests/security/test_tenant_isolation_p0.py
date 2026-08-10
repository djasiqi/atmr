"""Tests Lot 0 P0 — isolation tenant (SEC-04 / SEC-05)."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest


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


def _auth_headers(client, email, password="SecurePass1!"):
    resp = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
        headers={
            "Content-Type": "application/json",
            "X-Requested-With": "Expo",
        },
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    token = data.get("access_token") or data.get("token")
    assert token
    return {"Authorization": f"Bearer {token}"}


class TestInvoicesListAuth:
    def test_anonymous_gets_401(self, client, company_a):
        company, _ = company_a
        resp = client.get(f"/api/v1/invoices/companies/{company.id}/invoices")
        assert resp.status_code in (401, 422)

    def test_company_a_cannot_access_company_b(self, client, company_a, company_b):
        _, user_a = company_a
        company_b_obj, _ = company_b
        headers = _auth_headers(client, user_a.email)
        resp = client.get(
            f"/api/v1/invoices/companies/{company_b_obj.id}/invoices",
            headers=headers,
        )
        assert resp.status_code == 403

    def test_company_a_can_access_own(self, client, company_a):
        company, user_a = company_a
        headers = _auth_headers(client, user_a.email)
        resp = client.get(
            f"/api/v1/invoices/companies/{company.id}/invoices",
            headers=headers,
        )
        assert resp.status_code == 200

    def test_admin_cross_tenant_allowed_and_audited(
        self, client, company_a, admin_user
    ):
        company, _ = company_a
        headers = _auth_headers(client, admin_user.email)
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
    def test_company_b_blocked(self, client, company_a, company_b):
        company_a_obj, _ = company_a
        _, user_b = company_b
        headers = _auth_headers(client, user_b.email)
        resp = client.get(
            f"/api/v1/invoices/companies/{company_a_obj.id}/invoices/debug",
            headers=headers,
        )
        assert resp.status_code == 403


class TestExportPaymentsIdor:
    def test_company_b_blocked(self, client, company_a, company_b):
        company_a_obj, _ = company_a
        _, user_b = company_b
        headers = _auth_headers(client, user_b.email)
        resp = client.get(
            f"/api/v1/invoices/companies/{company_a_obj.id}/exports/payments.csv"
            f"?year=2026&month=1",
            headers=headers,
        )
        assert resp.status_code == 403
