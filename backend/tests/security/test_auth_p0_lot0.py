"""Tests Lot 0 P0 — passwordless hors dev + change-password / token_version."""

from __future__ import annotations

import uuid


class TestPasswordlessDisabledOutsideDev:
    def test_passwordless_404_in_production(self, client, app):
        app.config["ENVIRONMENT"] = "production"
        resp = client.post(
            "/api/v1/auth/passwordless/otp/request",
            json={"channel": "email", "identifier": "x@example.com"},
        )
        assert resp.status_code == 404

        resp2 = client.post(
            "/api/v1/auth/passwordless/otp/verify",
            json={"otp_session_id": "otp_x", "code": "123456"},
        )
        assert resp2.status_code == 404


class TestPasswordlessDebugCode:
    def _ensure_user(self, db, email: str):
        from models import User
        from models.enums import UserRole

        user = User(
            username=email.split("@")[0],
            email=email,
            role=UserRole.CLIENT,
        )
        user.set_password("SecurePass1!")
        db.session.add(user)
        db.session.commit()
        return user

    def test_no_debug_code_when_flag_false(self, client, app, db, monkeypatch):
        app.config["ENVIRONMENT"] = "development"
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("PASSWORDLESS_DEBUG_CODE", "false")
        email = f"debug-off-{uuid.uuid4().hex[:8]}@example.com"
        self._ensure_user(db, email)
        resp = client.post(
            "/api/v1/auth/passwordless/otp/request",
            json={"channel": "email", "identifier": email},
        )
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert "debug_code" not in data

    def test_debug_code_when_flag_true(self, client, app, db, monkeypatch):
        app.config["ENVIRONMENT"] = "development"
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("PASSWORDLESS_DEBUG_CODE", "true")
        email = f"debug-on-{uuid.uuid4().hex[:8]}@example.com"
        self._ensure_user(db, email)
        resp = client.post(
            "/api/v1/auth/passwordless/otp/request",
            json={"channel": "email", "identifier": email},
        )
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
        assert "debug_code" in data
        assert data["debug_code"]


class TestResetPasswordByPublicIdGone:
    def test_always_410(self, client):
        resp = client.post(
            "/api/v1/auth/reset-password/any-public-id",
            json={"new_password": "NewSecurePass1!"},
        )
        assert resp.status_code == 410
        data = resp.get_json()
        assert data.get("error") == "endpoint_removed"


class TestChangePasswordTokenVersion:
    def test_old_access_token_rejected_after_change(self, client, db):
        from models import Company, User
        from models.enums import UserRole

        suffix = uuid.uuid4().hex[:10]
        user = User(
            username=f"tv_user_{suffix}",
            email=f"tv_user_{suffix}@example.com",
            role=UserRole.COMPANY,
        )
        # set_password écrase force_password_change — passer force_change=True
        user.set_password("OldSecurePass1!", force_change=True)
        db.session.add(user)
        db.session.flush()
        db.session.add(Company(name=f"TV Co {suffix}", user_id=user.id))
        db.session.commit()

        login = client.post(
            "/api/v1/auth/login",
            json={"email": user.email, "password": "OldSecurePass1!"},
        )
        assert login.status_code == 200, login.get_json()
        token = login.get_json().get("access_token") or login.get_json().get("token")
        assert token
        headers = {"Authorization": f"Bearer {token}"}

        # Sanity: le token force-reset peut appeler change-password
        change = client.post(
            "/api/v1/auth/change-password",
            json={
                "new_password": "BrandNewSecure1!",
                "confirm_password": "BrandNewSecure1!",
            },
            headers=headers,
        )
        assert change.status_code == 200, change.get_json()

        # Ancien access token : rejeté (token_version mismatch)
        me = client.get("/api/v1/auth/me", headers=headers)
        assert me.status_code in (401, 422)
