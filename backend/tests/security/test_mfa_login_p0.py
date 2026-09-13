"""P0-03 — le mot de passe seul n'émet jamais de JWT métier si TOTP/enrollment requis."""

from __future__ import annotations

from models.enums import InstitutionRole, UserRole
from models.user import User


def _login(client, user, password: str = "password123", headers=None):
    return client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": password},
        headers=headers or {"X-Requested-With": "Expo"},
    )


def _make_user(db, *, role, totp_enabled=False, institution_role=None, suffix="u"):
    import uuid

    user = User()
    user.username = f"mfa-{suffix}-{uuid.uuid4().hex[:8]}"
    user.email = f"{user.username}@example.com"
    user.role = role
    user.public_id = str(uuid.uuid4())
    user.institution_role = institution_role
    user.totp_enabled = totp_enabled
    user.totp_secret_encrypted = "secret-chiffre" if totp_enabled else None
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    db.session.refresh(user)
    return user


def test_pyotp_is_available_for_totp_service():
    """Régression : setup/challenge ne doivent pas 500 faute de pyotp."""
    import pyotp

    assert hasattr(pyotp, "TOTP")


def test_totp_enabled_client_gets_challenge_not_jwt(client, db):
    user = _make_user(db, role=UserRole.CLIENT, totp_enabled=True, suffix="cli")
    resp = _login(client, user)
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["mfa_required"] is True
    assert data["mfa_purpose"] == "2fa_challenge"
    assert data.get("temp_token")
    assert "access_token" not in data
    assert "token" not in data
    assert "refresh_token" not in data


def test_totp_enabled_driver_gets_challenge_not_jwt(client, db):
    user = _make_user(db, role=UserRole.DRIVER, totp_enabled=True, suffix="drv")
    resp = _login(client, user)
    assert resp.status_code == 202
    assert resp.get_json()["mfa_purpose"] == "2fa_challenge"
    assert "token" not in resp.get_json()


def test_admin_without_totp_must_enroll(client, db):
    user = _make_user(db, role=UserRole.ADMIN, totp_enabled=False, suffix="adm")
    resp = _login(client, user)
    assert resp.status_code == 202
    data = resp.get_json()
    assert data["mfa_purpose"] == "mfa_enroll"
    assert data.get("temp_token")
    assert "token" not in data


def test_institution_admin_without_totp_must_enroll(client, db):
    user = _make_user(
        db,
        role=UserRole.INSTITUTION,
        totp_enabled=False,
        institution_role=InstitutionRole.ADMIN.value,
        suffix="iadm",
    )
    resp = _login(client, user)
    assert resp.status_code == 202
    assert resp.get_json()["mfa_purpose"] == "mfa_enroll"


def test_institution_requester_without_totp_gets_jwt(client, db):
    user = _make_user(
        db,
        role=UserRole.INSTITUTION,
        totp_enabled=False,
        institution_role=InstitutionRole.REQUESTER.value,
        suffix="req",
    )
    resp = _login(client, user)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("token") or data.get("access_token")
    assert data.get("refresh_token")


def test_client_without_totp_unchanged(client, sample_user):
    sample_user.role = UserRole.CLIENT
    resp = _login(client, sample_user)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("token") or data.get("access_token")


def test_challenge_issues_jwt_after_valid_code(client, app, db, monkeypatch):
    user = _make_user(db, role=UserRole.CLIENT, totp_enabled=True, suffix="ch")
    login = _login(client, user)
    assert login.status_code == 202
    temp = login.get_json()["temp_token"]

    monkeypatch.setattr(
        "security.totp_service.verify_totp_code", lambda _secret, code: code == "123456"
    )
    challenge = client.post(
        "/api/v1/auth/totp/challenge",
        json={"temp_token": temp, "code": "123456"},
        headers={"X-Requested-With": "Expo"},
    )
    assert challenge.status_code == 200
    data = challenge.get_json()
    assert data.get("token") or data.get("access_token")
    assert data.get("refresh_token")


def test_mfa_temp_token_rejected_on_me(client, db):
    user = _make_user(db, role=UserRole.ADMIN, totp_enabled=False, suffix="blk")
    login = _login(client, user)
    token = login.get_json()["temp_token"]
    me = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert me.status_code in (401, 422)


def test_privileged_cannot_self_disable_totp(client, app, db, monkeypatch):
    monkeypatch.setenv("SECURITY_2FA_ENABLED", "true")
    user = _make_user(db, role=UserRole.ADMIN, totp_enabled=True, suffix="dis")
    with app.app_context():
        from flask_jwt_extended import create_access_token

        token = create_access_token(
            identity=str(user.public_id),
            fresh=True,
            additional_claims={"role": user.role.value, "aud": "atmr-api"},
        )
    resp = client.post(
        "/api/v1/auth/totp/disable",
        json={"password": "password123"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403
    assert resp.get_json().get("error_code") == "mfa_disable_forbidden"


def test_enroll_setup_then_verify_returns_challenge(client, db, monkeypatch):
    monkeypatch.setattr(
        "security.totp_service.generate_totp_secret",
        lambda _label: {
            "secret_encrypted": "secret-chiffre",
            "provisioning_uri": "otpauth://test",
            "qr_code_base64": "abc",
            "secret_display": "AAAA",
        },
    )
    monkeypatch.setattr(
        "security.totp_service.verify_totp_code", lambda _secret, code: code == "123456"
    )
    monkeypatch.setattr(
        "security.totp_service.generate_recovery_codes",
        lambda: (["12345678"], '["h1"]'),
    )
    user = _make_user(db, role=UserRole.ADMIN, totp_enabled=False, suffix="enr")
    login = _login(client, user)
    temp = login.get_json()["temp_token"]
    headers = {"Authorization": f"Bearer {temp}"}

    setup = client.post("/api/v1/auth/totp/setup", json={}, headers=headers)
    assert setup.status_code == 200

    verify = client.post(
        "/api/v1/auth/totp/verify",
        json={"code": "123456"},
        headers=headers,
    )
    assert verify.status_code == 200
    data = verify.get_json()
    assert data["mfa_purpose"] == "2fa_challenge"
    assert data.get("temp_token")
    assert "token" not in data
    db.session.refresh(user)
    assert user.totp_enabled is True


def test_temp_token_cannot_refresh_into_business_jwt(client, db):
    user = _make_user(db, role=UserRole.CLIENT, totp_enabled=True, suffix="ref")
    login = _login(client, user)
    temp = login.get_json()["temp_token"]
    refresh = client.post(
        "/api/v1/auth/refresh-token",
        json={"refresh_token": temp},
        headers={"X-Requested-With": "Expo"},
    )
    assert refresh.status_code in (400, 401)
    body = refresh.get_json() or {}
    assert "access_token" not in body
    assert "token" not in body


def test_admin_disable_requires_totp_not_password_session(client, app, db):
    actor = _make_user(db, role=UserRole.ADMIN, totp_enabled=False, suffix="bg1")
    target = _make_user(db, role=UserRole.CLIENT, totp_enabled=True, suffix="bg2")
    with app.app_context():
        from flask_jwt_extended import create_access_token

        token = create_access_token(
            identity=str(actor.public_id),
            fresh=True,
            additional_claims={"role": actor.role.value, "aud": "atmr-api"},
        )
    resp = client.post(
        "/api/v1/auth/totp/admin-disable",
        json={
            "target_public_id": target.public_id,
            "reason": "incident-sev1",
            "code": "123456",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403
    assert resp.get_json().get("error_code") == "mfa_challenge_required"
    db.session.refresh(target)
    assert target.totp_enabled is True


def test_admin_disable_requires_valid_totp_code(client, app, db, monkeypatch):
    actor = _make_user(db, role=UserRole.ADMIN, totp_enabled=True, suffix="bg3")
    target = _make_user(db, role=UserRole.CLIENT, totp_enabled=True, suffix="bg4")
    monkeypatch.setattr(
        "security.totp_service.verify_totp_code", lambda _secret, code: code == "123456"
    )
    with app.app_context():
        from flask_jwt_extended import create_access_token

        token = create_access_token(
            identity=str(actor.public_id),
            fresh=True,
            additional_claims={"role": actor.role.value, "aud": "atmr-api"},
        )
    missing = client.post(
        "/api/v1/auth/totp/admin-disable",
        json={"target_public_id": target.public_id, "reason": "incident-sev1"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert missing.status_code == 401
    ok = client.post(
        "/api/v1/auth/totp/admin-disable",
        json={
            "target_public_id": target.public_id,
            "reason": "incident-sev1",
            "code": "123456",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert ok.status_code == 200
    db.session.refresh(target)
    assert target.totp_enabled is False
