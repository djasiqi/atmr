"""F-04 / F-05 — plan de contrôle ML : auth, kill-switch, tenant, inventaire."""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from services.infrastructure.ml_control_plane import (
    ML_CONTROL_PLANE_DISABLED_ERROR,
    ML_CONTROL_PLANE_PREFIXES,
)


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# 25 combinaisons méthode/route métier (hors Swagger)
ML_CONTROL_PLANE_ROUTES: list[tuple[str, str]] = [
    ("GET", "/api/feature-flags/status"),
    ("GET", "/api/feature-flags/runtime-status"),
    ("POST", "/api/feature-flags/ml/enable"),
    ("POST", "/api/feature-flags/ml/disable"),
    ("POST", "/api/feature-flags/ml/percentage"),
    ("POST", "/api/feature-flags/reset-stats"),
    ("GET", "/api/feature-flags/ml/health"),
    ("GET", "/api/shadow-mode/status"),
    ("GET", "/api/shadow-mode/stats"),
    ("GET", "/api/shadow-mode/predictions"),
    ("GET", "/api/shadow-mode/comparisons"),
    ("GET", "/api/shadow-mode/health"),
    ("GET", "/api/shadow-mode/companies"),
    ("POST", "/api/shadow-mode/session"),
    ("DELETE", "/api/shadow-mode/session"),
    ("GET", "/api/shadow-mode/reports/daily/1"),
    ("POST", "/api/shadow-mode/reports/daily/1"),
    ("GET", "/api/shadow-mode/reports/summary/1"),
    ("GET", "/api/shadow-mode/kpis/metrics/1"),
    ("GET", "/api/shadow-mode/kpis/export/1"),
    ("GET", "/api/ml-monitoring/metrics"),
    ("GET", "/api/ml-monitoring/daily"),
    ("GET", "/api/ml-monitoring/predictions"),
    ("GET", "/api/ml-monitoring/anomalies"),
    ("GET", "/api/ml-monitoring/summary"),
]


@pytest.fixture
def company_pair(db):
    from models import Company, User
    from models.enums import UserRole

    def _make(prefix: str):
        uname = _unique(prefix)
        user = User(
            username=uname,
            email=f"{uname}@example.com",
            role=UserRole.COMPANY,
        )
        user.set_password("SecurePass1!")
        db.session.add(user)
        db.session.flush()
        company = Company(name=f"Co {uname}", user_id=user.id)
        db.session.add(company)
        db.session.commit()
        return company, user

    return _make("coa"), _make("cob")


def _jwt_headers(client, user):
    from flask_jwt_extended import create_access_token

    claims = {
        "role": user.role.value,
        "aud": "atmr-api",
    }
    with client.application.app_context():
        token = create_access_token(
            identity=str(user.public_id), additional_claims=claims
        )
    return {"Authorization": f"Bearer {token}"}


def _request(client, method: str, path: str, **kwargs):
    return client.open(path, method=method, **kwargs)


@pytest.mark.usefixtures("app")
class TestMlControlPlaneAnonymous401:
    """Plan on, CSRF off (défaut tests) → anonyme = 401 exact."""

    def test_anonymous_matrix_401(self, client, monkeypatch):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        assert len(ML_CONTROL_PLANE_ROUTES) == 25
        for method, path in ML_CONTROL_PLANE_ROUTES:
            resp = _request(client, method, path)
            assert resp.status_code == 401, f"{method} {path} → {resp.status_code}"


class TestMlControlPlaneKillSwitch:
    def test_disabled_returns_503_exact(self, client, monkeypatch):
        monkeypatch.setenv("ML_CONTROL_PLANE_API_ENABLED", "false")
        for method, path in ML_CONTROL_PLANE_ROUTES:
            resp = _request(client, method, path)
            assert resp.status_code == 503, f"{method} {path}"
            body = resp.get_json() or {}
            assert body.get("error") == ML_CONTROL_PLANE_DISABLED_ERROR

    def test_invalid_value_fail_closed(self, client, monkeypatch):
        monkeypatch.setenv("ML_CONTROL_PLANE_API_ENABLED", "maybe")
        resp = client.get("/api/feature-flags/status")
        assert resp.status_code == 503
        assert (resp.get_json() or {}).get("error") == ML_CONTROL_PLANE_DISABLED_ERROR

    def test_disabled_handler_not_called(self, client, monkeypatch):
        monkeypatch.setenv("ML_CONTROL_PLANE_API_ENABLED", "false")
        with patch(
            "routes.feature_flags_routes.get_feature_flags_status"
        ) as spy:
            resp = client.get("/api/feature-flags/status")
            assert resp.status_code == 503
            spy.assert_not_called()


class TestMlControlPlaneRoles:
    def test_company_forbidden_on_feature_flags(
        self, client, auth_headers, monkeypatch
    ):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        resp = client.get("/api/feature-flags/status", headers=auth_headers)
        assert resp.status_code == 403

    def test_company_forbidden_on_ml_monitoring(
        self, client, auth_headers, monkeypatch
    ):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        resp = client.get("/api/ml-monitoring/predictions", headers=auth_headers)
        assert resp.status_code == 403

    def test_company_cross_tenant_shadow_reads(
        self, client, company_pair, monkeypatch
    ):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        (company_a, user_a), (company_b, _user_b) = company_pair
        headers = _jwt_headers(client, user_a)
        for path in (
            f"/api/shadow-mode/reports/daily/{company_b.id}",
            f"/api/shadow-mode/reports/summary/{company_b.id}",
            f"/api/shadow-mode/kpis/metrics/{company_b.id}",
        ):
            resp = client.get(path, headers=headers)
            assert resp.status_code == 403, path

    def test_company_own_daily_ok(self, client, company_pair, monkeypatch):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        (company_a, user_a), _ = company_pair
        headers = _jwt_headers(client, user_a)
        resp = client.get(
            f"/api/shadow-mode/reports/daily/{company_a.id}", headers=headers
        )
        assert resp.status_code == 200

    def test_admin_status_ok(self, client, admin_headers, monkeypatch):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        resp = client.get("/api/feature-flags/status", headers=admin_headers)
        assert resp.status_code == 200

    def test_admin_outside_whitelist_403(self, client, admin_headers, monkeypatch):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        monkeypatch.setenv("ADMIN_IP_WHITELIST", "198.51.100.10/32")
        resp = client.get(
            "/api/feature-flags/status",
            headers=admin_headers,
            environ_base={"REMOTE_ADDR": "203.0.113.99"},
        )
        assert resp.status_code == 403

    def test_xff_spoof_does_not_bypass_whitelist(
        self, client, admin_headers, monkeypatch
    ):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        monkeypatch.setenv("ADMIN_IP_WHITELIST", "198.51.100.10/32")
        resp = client.get(
            "/api/feature-flags/status",
            headers={
                **admin_headers,
                "X-Forwarded-For": "198.51.100.10, 203.0.113.99",
            },
            environ_base={"REMOTE_ADDR": "203.0.113.99"},
        )
        assert resp.status_code == 403

    def test_rejected_mutation_does_not_change_flags(
        self, client, auth_headers, monkeypatch
    ):
        from services.infrastructure.feature_flags import FeatureFlags

        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        FeatureFlags.set_ml_enabled(False)
        FeatureFlags.set_ml_traffic_percentage(0)
        resp = client.post(
            "/api/feature-flags/ml/enable",
            json={"percentage": 50},
            headers=auth_headers,
        )
        assert resp.status_code == 403
        assert FeatureFlags.is_ml_enabled() is False
        assert FeatureFlags.get_ml_traffic_percentage() == 0


class TestMlHealthVsKillSwitch:
    def test_degraded_health_not_kill_switch(self, client, admin_headers, monkeypatch):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        from services.infrastructure.feature_flags import FeatureFlags

        FeatureFlags.set_ml_enabled(False)
        resp = client.get("/api/feature-flags/ml/health", headers=admin_headers)
        assert resp.status_code == 503
        body = resp.get_json() or {}
        assert body.get("error") != ML_CONTROL_PLANE_DISABLED_ERROR
        assert "healthy" in body


class TestShadowBuildNoPersist:
    def test_get_daily_with_decisions_creates_no_files(
        self, client, company_pair, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        monkeypatch.setenv("RL_SHADOW_MODE_DIR", str(tmp_path))

        from routes.shadow_mode_routes import _shadow_manager_cache, get_shadow_manager

        _shadow_manager_cache.clear()
        (company_a, user_a), _ = company_pair
        manager = get_shadow_manager()
        company_key = str(company_a.id)
        manager.log_decision_comparison(
            company_id=company_key,
            booking_id="bk-1",
            human_decision={"driver_id": 1},
            rl_decision={"driver_id": 2},
            context={},
        )

        headers = _jwt_headers(client, user_a)
        resp = client.get(
            f"/api/shadow-mode/reports/daily/{company_a.id}", headers=headers
        )
        assert resp.status_code == 200
        assert (resp.get_json() or {}).get("total_decisions", 0) >= 1

        company_dir = Path(tmp_path) / company_key
        assert not company_dir.exists() or not any(company_dir.glob("report_*.json"))

    def test_get_daily_admin_no_persist(
        self, client, company_pair, admin_headers, monkeypatch, tmp_path
    ):
        monkeypatch.delenv("ML_CONTROL_PLANE_API_ENABLED", raising=False)
        monkeypatch.setenv("RL_SHADOW_MODE_DIR", str(tmp_path))

        from routes.shadow_mode_routes import _shadow_manager_cache, get_shadow_manager

        _shadow_manager_cache.clear()
        (company_a, _), _ = company_pair
        manager = get_shadow_manager()
        company_key = str(company_a.id)
        manager.log_decision_comparison(
            company_id=company_key,
            booking_id="bk-admin",
            human_decision={"driver_id": 1},
            rl_decision={"driver_id": 1},
            context={},
        )

        resp = client.get(
            f"/api/shadow-mode/reports/daily/{company_a.id}",
            headers=admin_headers,
        )
        assert resp.status_code == 200
        company_dir = Path(tmp_path) / company_key
        assert not company_dir.exists() or not any(company_dir.glob("report_*.json"))


class TestUrlMapInventory:
    def test_no_public_swagger_and_prefixes_known(self, app):
        """Pas de Swagger public ; chaque règle métier sous préfixe est connue."""
        structural_bases = {
            "/api/feature-flags/status",
            "/api/feature-flags/runtime-status",
            "/api/feature-flags/ml/enable",
            "/api/feature-flags/ml/disable",
            "/api/feature-flags/ml/percentage",
            "/api/feature-flags/reset-stats",
            "/api/feature-flags/ml/health",
            "/api/shadow-mode/status",
            "/api/shadow-mode/stats",
            "/api/shadow-mode/predictions",
            "/api/shadow-mode/comparisons",
            "/api/shadow-mode/health",
            "/api/shadow-mode/companies",
            "/api/shadow-mode/session",
            "/api/shadow-mode/reports/daily/",
            "/api/shadow-mode/reports/summary/",
            "/api/shadow-mode/kpis/metrics/",
            "/api/shadow-mode/kpis/export/",
            "/api/ml-monitoring/metrics",
            "/api/ml-monitoring/daily",
            "/api/ml-monitoring/predictions",
            "/api/ml-monitoring/anomalies",
            "/api/ml-monitoring/summary",
        }

        for rule in app.url_map.iter_rules():
            path = rule.rule
            if not any(
                path == p or path.startswith(f"{p}/") or path.startswith(p)
                for p in ML_CONTROL_PLANE_PREFIXES
            ):
                continue
            assert "swagger" not in path.lower()
            assert not path.rstrip("/").endswith("/docs")
            methods = {m for m in (rule.methods or set()) if m not in ("HEAD", "OPTIONS")}
            if not methods:
                continue
            matched = any(
                path == base or path.startswith(base)
                for base in structural_bases
            )
            assert matched, f"Route inconnue sous plan ML: {sorted(methods)} {path}"


class TestBootWhitelistValidation:
    def test_required_empty_fails(self, monkeypatch):
        from app import validate_required_env_vars

        monkeypatch.setenv("ADMIN_IP_WHITELIST_REQUIRED", "true")
        monkeypatch.setenv("ADMIN_IP_WHITELIST", "")
        monkeypatch.setenv("JWT_SECRET_KEY", "x" * 32)
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("SOCKETIO_CORS_ORIGINS", "https://app.lirie.ch")
        monkeypatch.setenv("PDF_BASE_URL", "https://api.lirie.ch")
        with pytest.raises(RuntimeError, match="ADMIN_IP_WHITELIST"):
            validate_required_env_vars("production")

    def test_required_garbage_fails(self, monkeypatch):
        from app import validate_required_env_vars

        monkeypatch.setenv("ADMIN_IP_WHITELIST_REQUIRED", "true")
        monkeypatch.setenv("ADMIN_IP_WHITELIST", "not-an-ip")
        monkeypatch.setenv("JWT_SECRET_KEY", "x" * 32)
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
        monkeypatch.setenv("SOCKETIO_CORS_ORIGINS", "https://app.lirie.ch")
        monkeypatch.setenv("PDF_BASE_URL", "https://api.lirie.ch")
        with pytest.raises(RuntimeError, match="ADMIN_IP_WHITELIST"):
            validate_required_env_vars("production")


class TestCsrfStrictSeparate:
    def test_post_without_csrf_token_returns_403(self, monkeypatch):
        """CSRF strict hors app TESTING : POST sans jeton → 403 exact."""
        from flask import Flask

        from services.infrastructure.ml_control_plane import (
            register_ml_control_plane_kill_switch,
        )
        from services.security.csrf import setup_csrf_protection

        monkeypatch.setenv("ML_CONTROL_PLANE_API_ENABLED", "true")
        monkeypatch.setenv("CSRF_ENABLED", "true")
        monkeypatch.setenv("CSRF_SECRET_KEY", "csrf-test-secret-key-32chars!!")

        mini = Flask(__name__)
        mini.config["TESTING"] = False
        mini.config["CSRF_ENABLED"] = True
        mini.config["WTF_CSRF_ENABLED"] = True

        register_ml_control_plane_kill_switch(mini)
        setup_csrf_protection(mini)

        @mini.route("/api/feature-flags/ml/disable", methods=["POST"])
        def _dummy_disable():
            return {"ok": True}, 200

        client = mini.test_client()
        resp = client.post(
            "/api/feature-flags/ml/disable",
            json={},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 403
        body = resp.get_json() or {}
        assert "CSRF" in (body.get("error") or "")
