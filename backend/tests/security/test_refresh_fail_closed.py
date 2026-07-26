"""Lot 1-C : refresh fail-closed — Redis/DB → 503 ; absent → 401."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from flask_jwt_extended import create_refresh_token
from sqlalchemy.exc import OperationalError

from security.refresh_token_service import (
    RefreshStoreUnavailableError,
    is_token_revoked,
    refresh_fail_closed_enabled,
)


@pytest.fixture
def fail_closed(monkeypatch):
    monkeypatch.setenv("REFRESH_FAIL_CLOSED", "true")


class TestRefreshFailClosedFlag:
    def test_flag_on(self, app, fail_closed):
        with app.app_context():
            assert refresh_fail_closed_enabled() is True


class TestIsTokenRevokedFailClosed:
    def test_absent_is_revoked(self, app, db, fail_closed):
        with app.app_context():
            assert is_token_revoked("not-a-real-jwt-token-xyz") is True

    def test_db_error_raises_unavailable(self, app, fail_closed):
        with (
            app.app_context(),
            patch("security.refresh_token_service.RefreshToken.query") as q,
        ):
            q.filter_by.side_effect = OperationalError("stmt", {}, Exception())
            with pytest.raises(RefreshStoreUnavailableError):
                is_token_revoked("any-token")


class TestRefreshEndpoint:
    def test_missing_cookie_web_401(self, client, fail_closed):
        resp = client.post("/api/v1/auth/refresh-token", json={})
        assert resp.status_code == 401

    def test_mobile_cookies_ignored_without_body_401(
        self, client, sample_user, fail_closed
    ):
        with client.application.app_context():
            rt = create_refresh_token(
                identity=str(sample_user.public_id),
                additional_claims={"aud": "atmr-api"},
            )
        client.set_cookie("refresh_token", rt)
        resp = client.post(
            "/api/v1/auth/refresh-token",
            json={},
            headers={"X-Requested-With": "Expo"},
        )
        assert resp.status_code == 401

    def test_store_unavailable_503(self, client, sample_user, fail_closed):
        login = client.post(
            "/api/v1/auth/login",
            json={"email": sample_user.email, "password": "password123"},
            headers={"X-Requested-With": "Expo"},
        )
        assert login.status_code == 200
        rt = login.get_json()["refresh_token"]
        with patch("routes.auth.RefreshTokenService") as svc_cls:
            instance = svc_cls.return_value
            instance.is_token_valid.side_effect = RefreshStoreUnavailableError(
                "redis_unavailable"
            )
            instance.store_token = lambda *a, **k: None
            instance.touch_token_score = lambda *a, **k: None
            instance.limit_active_tokens = lambda *a, **k: None
            resp = client.post(
                "/api/v1/auth/refresh-token",
                json={"refresh_token": rt},
                headers={"X-Requested-With": "Expo"},
            )
        assert resp.status_code == 503
