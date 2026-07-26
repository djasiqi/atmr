"""Lot 1-E : séparation web cookies / mobile Bearer + refresh JSON."""

from __future__ import annotations

import pytest


class TestWebMobileAuthSplit:
    def test_mobile_refresh_with_trapped_cookies_and_json_body(
        self, client, sample_user
    ):
        login = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )
        assert login.status_code == 200
        good = login.get_json()["refresh_token"]
        # Cookie piégé — ignoré pour l'auth mobile
        client.set_cookie("refresh_token", "trapped-refresh-token")
        client.set_cookie("access_token", "trapped-access")

        resp = client.post(
            "/api/v1/auth/refresh-token",
            json={"refresh_token": good},
            headers={"X-Requested-With": "Expo"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("access_token")
        assert data.get("refresh_token")

    def test_mobile_cookies_only_without_body_401(self, client, sample_user):
        login = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )
        assert login.status_code == 200
        rt = login.get_json()["refresh_token"]
        client.set_cookie("refresh_token", rt)
        resp = client.post(
            "/api/v1/auth/refresh-token",
            json={},
            headers={"X-Requested-With": "Expo"},
        )
        assert resp.status_code == 401

    def test_web_login_omits_tokens_in_json(self, client, sample_user):
        resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": sample_user.email,
                "password": "password123",
            },
            headers={"Origin": "http://localhost"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "token" not in data
        assert "refresh_token" not in data
        assert "access_token" not in data
        set_cookie = ", ".join(resp.headers.getlist("Set-Cookie"))
        assert "access_token=" in set_cookie
        assert "refresh_token=" in set_cookie

    def test_mobile_login_returns_tokens_json(self, client, sample_user):
        resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": sample_user.email,
                "password": "password123",
            },
            headers={
                "Origin": "http://localhost",
                "X-Requested-With": "Expo",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("token") or data.get("access_token")
        assert data.get("refresh_token")

    def test_mobile_login_skips_origin_check_in_prod(
        self, client, sample_user, monkeypatch
    ):
        """Lot 1-E : Bearer mobile ne doit pas échouer en missing_origin."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("LOGIN_ALLOWED_ORIGINS", "https://www.lirie.ch")
        monkeypatch.setitem(client.application.config, "TESTING", False)
        resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": sample_user.email,
                "password": "password123",
            },
            headers={"X-Requested-With": "Expo"},
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data.get("token") or data.get("access_token")
        assert data.get("refresh_token")

    def test_web_login_missing_origin_forbidden_in_prod(
        self, client, sample_user, monkeypatch
    ):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("LOGIN_ALLOWED_ORIGINS", "https://www.lirie.ch")
        monkeypatch.setitem(client.application.config, "TESTING", False)
        resp = client.post(
            "/api/v1/auth/login",
            json={
                "email": sample_user.email,
                "password": "password123",
            },
        )
        assert resp.status_code == 403
        assert resp.get_json().get("error") == "missing_origin"
