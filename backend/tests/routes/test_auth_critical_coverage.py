"""Couverture d'intégration ciblée des routes critiques d'authentification."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from flask_jwt_extended import create_access_token

from routes import auth


def _fresh_headers(app, user):
    """Crée un JWT frais pour les opérations sensibles."""
    with app.app_context():
        token = create_access_token(
            identity=str(user.public_id),
            fresh=True,
            additional_claims={"role": user.role.value, "aud": "atmr-api"},
        )
    return {"Authorization": f"Bearer {token}"}


def test_routes_totp_desactivees(client, app, sample_user, monkeypatch):
    monkeypatch.setenv("SECURITY_2FA_ENABLED", "false")
    headers = _fresh_headers(app, sample_user)
    calls = [
        ("post", "/api/v1/auth/totp/setup", {}),
        ("post", "/api/v1/auth/totp/verify", {"code": "123456"}),
        ("post", "/api/v1/auth/totp/disable", {"password": "password123"}),
        ("post", "/api/v1/auth/totp/recovery-codes", {"code": "123456"}),
        ("post", "/api/v1/auth/totp/challenge", {"temp_token": "x", "code": "123456"}),
    ]
    for method, url, payload in calls:
        response = getattr(client, method)(url, json=payload, headers=headers)
        assert response.status_code == 403


def test_cycle_totp_active(client, app, db, sample_user, monkeypatch):
    monkeypatch.setenv("SECURITY_2FA_ENABLED", "true")
    monkeypatch.setattr(
        auth.user_repo, "find_by_public_id", lambda _public_id: sample_user
    )
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
        lambda: (["12345678", "87654321"], '["h1","h2"]'),
    )
    monkeypatch.setattr(
        "shared.audit_helpers.audit_log", lambda *_args, **_kwargs: None
    )
    headers = _fresh_headers(app, sample_user)

    setup = client.post("/api/v1/auth/totp/setup", json={}, headers=headers)
    assert setup.status_code == 200
    verify = client.post(
        "/api/v1/auth/totp/verify", json={"code": "123456"}, headers=headers
    )
    assert verify.status_code == 200
    status = client.get("/api/v1/auth/totp/status", headers=headers)
    assert status.status_code == 200
    assert status.get_json()["enabled"] is True
    recovery = client.post(
        "/api/v1/auth/totp/recovery-codes",
        json={"code": "123456"},
        headers=headers,
    )
    assert recovery.status_code == 200
    disable = client.post(
        "/api/v1/auth/totp/disable",
        json={"password": "password123"},
        headers=headers,
    )
    assert disable.status_code == 200
    db.session.flush()


def test_challenge_totp_token_invalide(client, monkeypatch):
    monkeypatch.setenv("SECURITY_2FA_ENABLED", "true")
    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda _token: {"purpose": "autre", "jti": "j1", "sub": "u1"},
    )
    response = client.post(
        "/api/v1/auth/totp/challenge",
        json={"temp_token": "temp", "code": "123456"},
    )
    assert response.status_code == 401


def test_fresh_token_bon_et_mauvais_mot_de_passe(client, app, sample_user, monkeypatch):
    monkeypatch.setattr(
        auth.user_repo, "find_by_public_id", lambda _public_id: sample_user
    )
    monkeypatch.setattr(
        auth.user_repo, "find_model_by_email", lambda _email: sample_user
    )
    headers = _fresh_headers(app, sample_user)
    bad = client.post(
        "/api/v1/auth/fresh-token",
        json={"password": "incorrect"},
        headers=headers,
    )
    assert bad.status_code == 401
    good = client.post(
        "/api/v1/auth/fresh-token",
        json={"password": "password123"},
        headers={**headers, "X-Requested-With": "Expo"},
    )
    assert good.status_code == 200
    assert good.get_json()["access_token"]


def test_sessions_liste_et_revoque_les_autres(client, app, sample_user, monkeypatch):
    session_a = SimpleNamespace(
        token_hash="a",
        is_revoked=False,
        revoked_at=None,
        revoked_reason=None,
        serialize_masked=lambda **_kwargs: {"id": 1, "is_current": False},
    )
    session_b = SimpleNamespace(
        token_hash="b",
        is_revoked=False,
        revoked_at=None,
        revoked_reason=None,
        serialize_masked=lambda **_kwargs: {"id": 2, "is_current": False},
    )
    monkeypatch.setattr(
        auth.user_repo, "find_by_public_id", lambda _public_id: sample_user
    )
    monkeypatch.setattr(
        auth, "get_user_active_sessions", lambda _user_id: [session_a, session_b]
    )
    monkeypatch.setattr(
        "shared.audit_helpers.audit_log", lambda *_args, **_kwargs: None
    )
    headers = _fresh_headers(app, sample_user)

    listed = client.get("/api/v1/auth/sessions", headers=headers)
    assert listed.status_code == 200
    assert listed.get_json()["count"] == 2
    revoked = client.post(
        "/api/v1/auth/sessions/revoke-others", json={}, headers=headers
    )
    assert revoked.status_code == 200
    assert revoked.get_json()["revoked_count"] == 2
    assert session_a.is_revoked is True


def test_revoque_toutes_sessions_admin(
    client, sample_user, sample_admin_user, admin_headers, monkeypatch
):
    monkeypatch.setattr(
        auth.user_repo,
        "find_by_id",
        lambda user_id: sample_user if user_id == sample_user.id else None,
    )
    monkeypatch.setattr(
        auth.user_repo, "find_by_public_id", lambda _public_id: sample_admin_user
    )
    monkeypatch.setattr(auth, "revoke_all_user_tokens", lambda **_kwargs: 3)
    monkeypatch.setattr(auth.AuditLogger, "log_action", lambda **_kwargs: None)
    response = client.post(
        f"/api/v1/auth/revoke-all-sessions/{sample_user.id}",
        json={},
        headers=admin_headers,
    )
    assert response.status_code == 200
    assert response.get_json()["sessions_revoked"] == 3


def test_switch_context_valide_et_invalide(
    client, app, sample_user, sample_company, monkeypatch
):
    monkeypatch.setattr(
        auth, "_load_user_for_bootstrap", lambda _public_id: sample_user
    )
    monkeypatch.setattr(auth, "_prepare_user_for_bootstrap", lambda user: user)
    monkeypatch.setattr(auth, "_save_active_context", lambda *_args: None)
    monkeypatch.setattr(auth, "_get_saved_active_context", lambda _public_id: None)
    headers = _fresh_headers(app, sample_user)
    invalid = client.post(
        "/api/v1/auth/switch-context",
        json={"target_context_id": "client:absent"},
        headers=headers,
    )
    assert invalid.status_code == 403
    valid = client.post(
        "/api/v1/auth/switch-context",
        json={"target_context_id": f"company:{sample_company.id}"},
        headers=headers,
    )
    assert valid.status_code == 200


def test_parcours_reservation_invitee(client, app, monkeypatch):
    monkeypatch.setattr(auth, "redis_client", None)
    auth._PUBLIC_PRE_REQUEST_CACHE.clear()
    monkeypatch.setattr(
        auth,
        "compute_public_guest_booking_price",
        lambda **_kwargs: {
            "ok": True,
            "amount": 42.5,
            "currency": "CHF",
            "distance_meters": 5000,
            "duration_seconds": 900,
            "pricing_status": "confirmed",
        },
    )
    payload = {
        "departure": "Lausanne",
        "destination": "Pully",
        "date": "2030-01-01",
        "pickup_time": "10:30",
    }
    preview = client.post("/api/v1/auth/public/guest-booking/preview", json=payload)
    assert preview.status_code == 200
    created = client.post("/api/v1/auth/public/guest-booking/create", json=payload)
    assert created.status_code == 201
    data = created.get_json()
    status = client.get(
        "/api/v1/auth/public/guest-booking/status",
        query_string={"token": data["status_token"]},
    )
    assert status.status_code == 200

    monkeypatch.setattr("services.saferpay.config.saferpay_configured", lambda: True)
    monkeypatch.setattr(
        "services.guest_saferpay.initialize_guest_saferpay",
        lambda **_kwargs: {"redirect_url": "https://example.test/pay"},
    )
    initialized = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/initialize",
        json={
            "guest_booking_id": data["guest_booking_id"],
            "status_token": data["status_token"],
        },
    )
    assert initialized.status_code == 200

    monkeypatch.setattr(
        "services.guest_saferpay.promote_guest_booking_after_saferpay",
        lambda **_kwargs: {"status": "completed", "booking_id": 42},
    )
    asserted = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/assert",
        json={
            "guest_booking_id": data["guest_booking_id"],
            "status_token": data["status_token"],
        },
    )
    assert asserted.status_code == 200
    assert asserted.get_json()["payment_status"] == "paid"


def test_passwordless_demande_et_verification(client, app, sample_user, monkeypatch):
    monkeypatch.setitem(app.config, "ENVIRONMENT", "development")
    monkeypatch.setenv("PASSWORDLESS_DEBUG_CODE", "true")
    monkeypatch.setattr(auth, "redis_client", None)
    auth._PUBLIC_PRE_REQUEST_CACHE.clear()
    monkeypatch.setattr(auth, "_create_passwordless_otp_code", lambda: "123456")
    monkeypatch.setattr(auth, "_check_user_profile_active", lambda _user: (True, None))
    response = client.post(
        "/api/v1/auth/passwordless/otp/request",
        json={"channel": "email", "identifier": sample_user.email},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["debug_code"] == "123456"
    verified = client.post(
        "/api/v1/auth/passwordless/otp/verify",
        json={"otp_session_id": body["otp_session_id"], "code": "123456"},
    )
    assert verified.status_code == 200
    assert verified.get_json()["auth_mode"] == "passwordless_otp"


def test_mot_de_passe_oublie_et_ancien_reset(client, sample_user, monkeypatch):
    monkeypatch.setattr(auth.user_repo, "find_by_email", lambda _email: sample_user)
    monkeypatch.setattr(
        auth,
        "send_email_notification",
        lambda **_kwargs: {"ok": True},
    )
    forgot = client.post(
        "/api/v1/auth/forgot-password", json={"email": sample_user.email}
    )
    assert forgot.status_code == 200
    removed = client.post(
        f"/api/v1/auth/reset-password/{sample_user.public_id}",
        json={"new_password": "password123"},
    )
    assert removed.status_code == 410


def test_activation_erreurs_csrf_login_test_et_invitation(
    client, sample_user, monkeypatch
):
    assert client.get("/api/v1/auth/activation/status").status_code == 400
    update_phone = client.post(
        "/api/v1/auth/activation/update-phone",
        json={"activation_session_id": "absente", "phone": "0791234567"},
    )
    assert update_phone.status_code in {400, 404}
    monkeypatch.setattr(auth, "generate_csrf_token", lambda **_kwargs: "csrf-test")
    csrf = client.get("/api/v1/auth/csrf-token")
    assert csrf.status_code == 200
    assert csrf.get_json()["csrf_token"] == "csrf-test"
    login_test = client.post(
        "/api/v1/auth/login-test",
        json={"email": sample_user.email, "password": "password123"},
    )
    assert login_test.status_code in {200, 404}
    assert client.get("/api/v1/auth/invite/token-invalide").status_code == 400
    assert (
        client.post(
            "/api/v1/auth/activate-account",
            json={"token": "", "password": "court"},
        ).status_code
        == 400
    )


def test_branches_erreur_reservation_invitee(client, monkeypatch):
    """Les erreurs publiques restent explicites sans dépendance externe."""
    assert (
        client.post("/api/v1/auth/public/guest-booking/preview", json={}).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/auth/public/guest-booking/preview",
            json={
                "departure": "A",
                "destination": "B",
                "date": "2030-01-01",
            },
        ).status_code
        == 400
    )
    monkeypatch.setattr(
        auth,
        "compute_public_guest_booking_price",
        lambda **_kwargs: {"ok": False, "error": "invalid_schedule"},
    )
    failed = client.post(
        "/api/v1/auth/public/guest-booking/preview",
        json={
            "departure": "A",
            "destination": "B",
            "date": "2030-01-01",
            "pickup_time": "10:00",
        },
    )
    assert failed.status_code == 400
    assert client.get("/api/v1/auth/public/guest-booking/status").status_code == 401
    assert (
        client.get(
            "/api/v1/auth/public/guest-booking/status",
            query_string={"token": "invalide"},
        ).status_code
        == 401
    )
    assert client.post(
        "/api/v1/auth/public/guest-booking/saferpay/initialize", json={}
    ).status_code in {401, 503}
    assert client.post(
        "/api/v1/auth/public/guest-booking/saferpay/assert", json={}
    ).status_code in {401, 503}


def test_branches_erreur_passwordless(client, app, monkeypatch):
    monkeypatch.setitem(app.config, "ENVIRONMENT", "production")
    assert (
        client.post("/api/v1/auth/passwordless/otp/request", json={}).status_code == 404
    )
    assert (
        client.post("/api/v1/auth/passwordless/otp/verify", json={}).status_code == 404
    )
    monkeypatch.setitem(app.config, "ENVIRONMENT", "development")
    monkeypatch.setattr(auth, "redis_client", None)
    auth._PUBLIC_PRE_REQUEST_CACHE.clear()
    unknown = client.post(
        "/api/v1/auth/passwordless/otp/request",
        json={"channel": "email", "identifier": "absent@example.test"},
    )
    assert unknown.status_code == 404
    expired = client.post(
        "/api/v1/auth/passwordless/otp/verify",
        json={"otp_session_id": "otp_absente", "code": "123456"},
    )
    assert expired.status_code == 410


def test_branches_erreur_totp(client, app, sample_user, monkeypatch):
    monkeypatch.setenv("SECURITY_2FA_ENABLED", "true")
    monkeypatch.setattr(
        auth.user_repo, "find_by_public_id", lambda _public_id: sample_user
    )
    headers = _fresh_headers(app, sample_user)
    sample_user.totp_secret_encrypted = None
    sample_user.totp_enabled = False
    assert (
        client.post(
            "/api/v1/auth/totp/verify", json={"code": "12"}, headers=headers
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/auth/totp/verify",
            json={"code": "123456"},
            headers=headers,
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/auth/totp/recovery-codes",
            json={"code": "123456"},
            headers=headers,
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/v1/auth/totp/disable",
            json={"password": "incorrect"},
            headers=headers,
        ).status_code
        == 401
    )
    assert client.post("/api/v1/auth/totp/challenge", json={}).status_code == 400
