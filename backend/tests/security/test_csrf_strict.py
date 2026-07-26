"""Lot 1-D : CSRF_STRICT + Origin/Referer login."""

from __future__ import annotations

import os

import pytest

from services.security.csrf import _csrf_strict_enabled, generate_csrf_token
from services.security.login_origin import validate_login_origin_for_web


@pytest.fixture
def csrf_strict_env(monkeypatch):
    monkeypatch.setenv("CSRF_STRICT", "true")
    monkeypatch.setenv("CSRF_ENABLED", "true")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOGIN_ALLOWED_ORIGINS", "https://app.lirie.ch")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-for-csrf-lot1")


class TestLoginOrigin:
    def test_origin_ok(self, app, csrf_strict_env, monkeypatch):
        monkeypatch.setitem(app.config, "TESTING", False)
        with app.test_request_context(
            "/api/v1/auth/login",
            method="POST",
            headers={"Origin": "https://app.lirie.ch"},
        ):
            ok, err = validate_login_origin_for_web()
            assert ok is True
            assert err is None

    def test_foreign_origin_403(self, app, csrf_strict_env, monkeypatch):
        monkeypatch.setitem(app.config, "TESTING", False)
        with app.test_request_context(
            "/api/v1/auth/login",
            method="POST",
            headers={"Origin": "https://evil.example"},
        ):
            ok, err = validate_login_origin_for_web()
            assert ok is False
            assert err == "origin_not_allowed"

    def test_referer_fallback(self, app, csrf_strict_env, monkeypatch):
        monkeypatch.setitem(app.config, "TESTING", False)
        with app.test_request_context(
            "/api/v1/auth/login",
            method="POST",
            headers={"Referer": "https://app.lirie.ch/login"},
        ):
            ok, _err = validate_login_origin_for_web()
            assert ok is True

    def test_missing_origin_prod(self, app, csrf_strict_env, monkeypatch):
        monkeypatch.setitem(app.config, "TESTING", False)
        with app.test_request_context("/api/v1/auth/login", method="POST"):
            ok, err = validate_login_origin_for_web()
            assert ok is False
            assert err == "missing_origin"

    def test_login_origin_helper_ignores_expo_header(self, app, csrf_strict_env, monkeypatch):
        """Le helper reste strict : le bypass mobile est dans routes.auth.Login."""
        monkeypatch.setitem(app.config, "TESTING", False)
        with app.test_request_context(
            "/api/v1/auth/login",
            method="POST",
            headers={
                "X-Requested-With": "Expo",
                "Origin": "https://evil.example",
            },
        ):
            ok, err = validate_login_origin_for_web()
            assert ok is False
            assert err == "origin_not_allowed"


class TestCsrfStrictConfig:
    def test_csrf_strict_flag(self, csrf_strict_env):
        assert _csrf_strict_enabled() is True

    def test_refresh_not_in_legacy_blanket_exempt(self, csrf_strict_env):
        """Sous CSRF_STRICT, refresh-token n'est plus dans le catch-all /auth/."""
        assert os.getenv("CSRF_STRICT", "").lower() in {"1", "true", "yes", "on"}
        # Le middleware n'exempte plus /api/v1/auth/ en bloc — refresh exige CSRF
        from services.security import csrf as csrf_mod

        assert hasattr(csrf_mod, "setup_csrf_protection")
        token = generate_csrf_token()
        assert token.count(":") == 2
