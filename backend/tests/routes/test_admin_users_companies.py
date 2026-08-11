"""Tests routes admin — users / companies + CAP refusée."""

from __future__ import annotations

import uuid

import pytest

from models import Company, User
from models.enums import UserRole
from models.platform_admin_permission_grant import PlatformAdminPermissionGrant
from services.admin_authz import CAP_PARTNERS_READ
from tests.routes.admin_route_fixtures import ADMIN_ENVIRON, admin_auth_headers


@pytest.fixture
def admin_users_world(db):
    suffix = uuid.uuid4().hex[:8]
    company_user = User()
    company_user.username = f"auco_{suffix}"
    company_user.email = f"auco_{suffix}@test.ch"
    company_user.role = UserRole.company
    company_user.public_id = str(uuid.uuid4())
    company_user.set_password("password123", force_change=False)
    db.session.add(company_user)
    db.session.flush()

    company = Company()
    company.name = f"Admin Users Co {suffix}"
    company.address = "Rue 1"
    company.contact_email = company_user.email
    company.user_id = company_user.id
    company.is_approved = False
    db.session.add(company)
    db.session.commit()
    return {"company_user": company_user, "company": company}


class TestAdminUsersCompanies:
    def test_list_users_200(
        self, client, app, admin_route_env, make_admin_user, admin_users_world
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/users?page=1&per_page=20",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_list_users_cap_denied_403(
        self, client, app, make_admin_user, monkeypatch
    ):
        monkeypatch.setenv("ADMIN_IP_WHITELIST", "127.0.0.1/32")
        monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "true")
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/users",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 403
        body = resp.get_json() or {}
        assert body.get("error") == "forbidden"
        assert body.get("capability") == CAP_PARTNERS_READ

    def test_list_users_cap_granted_200(
        self, client, app, make_admin_user, db, monkeypatch
    ):
        monkeypatch.setenv("ADMIN_IP_WHITELIST", "127.0.0.1/32")
        monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "true")
        admin = make_admin_user()
        db.session.add(
            PlatformAdminPermissionGrant(
                user_id=admin.id,
                permission=CAP_PARTNERS_READ,
            )
        )
        db.session.commit()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/users",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_list_companies_200(
        self, client, app, admin_route_env, make_admin_user, monkeypatch
    ):
        """Évite le N+1 serialize de toutes les companies en DB de test."""
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)

        monkeypatch.setattr(
            "routes.admin.company_repo.find_all",
            lambda: [],
        )
        resp = client.get(
            "/api/v1/admin/companies",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200
        assert resp.get_json() == {"companies": []}

    def test_list_institutions_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/institutions",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_put_role_400(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.put(
            f"/api/v1/admin/users/{admin.id}/role",
            json={},
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 400

    def test_put_role_user_missing_400_or_404(
        self, client, app, admin_route_env, make_admin_user
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.put(
            "/api/v1/admin/users/999999/role",
            json={
                "role": "client",
                "expected_current_role": "company",
                "reason": "test coverage role transition missing user",
            },
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code in (400, 404)

    def test_company_approval_400(
        self, client, app, admin_route_env, make_admin_user, admin_users_world
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        cid = admin_users_world["company"].id
        resp = client.put(
            f"/api/v1/admin/companies/{cid}/approval",
            json={},
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 400

    def test_dispatch_status_400(
        self, client, app, admin_route_env, make_admin_user, admin_users_world
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        cid = admin_users_world["company"].id
        resp = client.put(
            f"/api/v1/admin/companies/{cid}/dispatch-status",
            json={},
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 400

    def test_reset_password_400(
        self, client, app, admin_route_env, make_admin_user, admin_users_world
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        uid = admin_users_world["company_user"].id
        resp = client.post(
            f"/api/v1/admin/users/{uid}/reset-password",
            json={},
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 400

    def test_users_401(self, client, admin_route_env):
        resp = client.get("/api/v1/admin/users", environ_base=ADMIN_ENVIRON)
        assert resp.status_code == 401
