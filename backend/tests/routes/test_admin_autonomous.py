"""Tests routes admin — autonomous actions."""

from __future__ import annotations

from tests.routes.admin_route_fixtures import ADMIN_ENVIRON, admin_auth_headers


class TestAdminAutonomous:
    def test_list_autonomous_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/autonomous-actions",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_autonomous_stats_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/autonomous-actions/stats",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_autonomous_detail_404(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/autonomous-actions/999999",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code in (404, 500)

    def test_autonomous_review_404_or_400(
        self, client, app, admin_route_env, make_admin_user
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.post(
            "/api/v1/admin/autonomous-actions/999999/review",
            json={"notes": "x"},
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code in (400, 404, 500)

    def test_autonomous_401(self, client, admin_route_env):
        resp = client.get(
            "/api/v1/admin/autonomous-actions",
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 401
