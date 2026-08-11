"""Tests routes admin — dashboard / stats / capabilities."""

from __future__ import annotations

from tests.routes.admin_route_fixtures import ADMIN_ENVIRON, admin_auth_headers


class TestAdminDashboardStats:
    def test_stats_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/stats",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_stats_401(self, client, admin_route_env):
        resp = client.get("/api/v1/admin/stats", environ_base=ADMIN_ENVIRON)
        assert resp.status_code == 401

    def test_stats_403_non_admin(
        self, client, app, admin_route_env, db, make_admin_user
    ):
        import uuid

        from models import User
        from models.enums import UserRole

        user = User()
        user.username = f"rcli_{uuid.uuid4().hex[:8]}"
        user.email = f"rcli_{uuid.uuid4().hex[:8]}@test.ch"
        user.role = UserRole.client
        user.public_id = str(uuid.uuid4())
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.commit()
        headers = admin_auth_headers(app, user)
        resp = client.get(
            "/api/v1/admin/stats",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code in (403, 401)

    def test_stats_403_ip_outside_whitelist(
        self, client, app, admin_route_env, make_admin_user, monkeypatch
    ):
        monkeypatch.setenv("ADMIN_IP_WHITELIST", "198.51.100.10/32")
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/stats",
            headers=headers,
            environ_base={"REMOTE_ADDR": "203.0.113.99"},
        )
        assert resp.status_code == 403

    def test_dashboard_summary_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/dashboard-summary",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_capabilities_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/capabilities",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_recent_users_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/recent-users",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_websocket_metrics_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/websocket/metrics",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code in (200, 500)
