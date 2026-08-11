"""Tests routes admin — rate-limit / redis / indicative-fare / optuna CAP."""

from __future__ import annotations

from services.admin_authz import CAP_LABS_EXECUTE
from tests.routes.admin_route_fixtures import ADMIN_ENVIRON, admin_auth_headers


class TestAdminOpsInfra:
    def test_rate_limit_stats_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/rate-limit/stats",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_rate_limit_config_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/rate-limit/config",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_redis_info_200(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/redis/info",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_indicative_fare_get(
        self, client, app, admin_route_env, make_admin_user
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/client-indicative-fare",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code in (200, 404)

    def test_indicative_fare_put_400_or_404(
        self, client, app, admin_route_env, make_admin_user
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.put(
            "/api/v1/admin/client-indicative-fare",
            json={"base_fare_chf": -1},
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code in (200, 400, 404)

    def test_optuna_cap_denied_403(self, client, app, make_admin_user, monkeypatch):
        monkeypatch.setenv("ADMIN_IP_WHITELIST", "127.0.0.1/32")
        monkeypatch.setenv("ADMIN_CAPABILITIES_ENFORCED", "true")
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.post(
            "/api/v1/admin/optuna/optimize",
            json={},
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 403
        body = resp.get_json() or {}
        assert body.get("capability") == CAP_LABS_EXECUTE

    def test_ops_401(self, client, admin_route_env):
        resp = client.get(
            "/api/v1/admin/redis/info",
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 401

    def test_rate_limit_flush(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.post(
            "/api/v1/admin/rate-limit/flush",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code in (200, 503, 500)

    def test_partners_organizations_200(
        self, client, app, admin_route_env, make_admin_user, monkeypatch
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        monkeypatch.setattr(
            "routes.admin.list_organizations_with_read_mode",
            lambda **_kwargs: {"items": [], "total": 0, "page": 1, "per_page": 50},
        )
        resp = client.get(
            "/api/v1/admin/partners/organizations",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 200

    def test_push_coverage_drivers(self, client, app, admin_route_env, make_admin_user):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/push-coverage/drivers?company_id=abc",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 400

    def test_saferpay_lookup_ref_too_short(
        self, client, app, admin_route_env, make_admin_user
    ):
        admin = make_admin_user()
        headers = admin_auth_headers(app, admin)
        resp = client.get(
            "/api/v1/admin/support/saferpay-payment-lookup?ref=ab",
            headers=headers,
            environ_base=ADMIN_ENVIRON,
        )
        assert resp.status_code == 400
