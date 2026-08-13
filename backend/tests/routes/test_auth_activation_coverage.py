"""Couverture ciblée des branches d'activation et de rotation JWT."""

from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from models import UserRole
from routes import auth


class _Query:
    def __init__(self, value):
        self.value = value

    def filter_by(self, **_kwargs):
        return self

    def filter(self, *_args, **_kwargs):
        return self

    def limit(self, _limit):
        return self

    def all(self):
        return self.value if isinstance(self.value, list) else []

    def first(self):
        if isinstance(self.value, list):
            return self.value[0] if self.value else None
        return self.value

    def get(self, _identifier):
        return self.value

    def populate_existing(self):
        return self

    def with_for_update(self):
        return self

    def one(self):
        return self.first()


def _activation_session(**overrides):
    values = {
        "id": 1,
        "activation_session_id": "session-test",
        "user_id": 7,
        "email_verified_at": None,
        "phone_verified_at": None,
        "consumed_at": None,
        "email_delivery_id": None,
        "email_token_hash": "hash",
        "email_token_expires_at": datetime.now(UTC) + timedelta(minutes=10),
        "email_delivery_status": "sent",
        "sms_code_hash": auth._hash_plain_value("123456"),
        "sms_expires_at": datetime.now(UTC) + timedelta(minutes=10),
        "sms_attempts": 0,
        "sms_locked_until": None,
        "last_email_sent_at": None,
        "last_sms_sent_at": None,
        "resend_count_email": 0,
        "resend_count_sms": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _activation_user(**overrides):
    values = {
        "id": 7,
        "public_id": "public-7",
        "username": "activation",
        "email": "activation@example.test",
        "phone": "+41791234567",
        "role": UserRole.client,
        "clients": [SimpleNamespace(is_active=False)],
        "account_status": "pending_activation",
        "driver_id": None,
        "institution_id": None,
        "institution_role": None,
        "token_version": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _patch_activation_models(monkeypatch, session, user):
    monkeypatch.setattr(
        auth,
        "ActivationSession",
        SimpleNamespace(query=_Query(session)),
    )
    monkeypatch.setattr(auth, "User", SimpleNamespace(query=_Query(user)))
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    monkeypatch.setattr(auth.db.session, "rollback", lambda: None)
    monkeypatch.setattr(auth, "_build_activation_status", lambda _session: {"ok": True})


def test_register_succes_email_sms(client, monkeypatch):
    payload = {
        "username": "nouveau",
        "email": "nouveau@example.test",
        "password": "MotDePasse123!",
        "first_name": "Jean",
        "last_name": "Test",
        "phone": "+41791234567",
        "address": "Rue du Test 1",
    }
    user = _activation_user(username="nouveau", email=payload["email"])
    result = SimpleNamespace(success=True, user=user, error=None, status_code=201)
    created_sessions = []

    class FakeActivationSession:
        def __init__(self):
            created_sessions.append(self)

    monkeypatch.setattr(auth, "validate_request", lambda *_args, **_kwargs: payload)
    monkeypatch.setattr(
        auth,
        "RegisterUserUseCase",
        lambda: SimpleNamespace(execute=lambda _data: result),
    )
    monkeypatch.setattr(
        auth,
        "Client",
        type("FakeClient", (), {"id": 17}),
    )
    monkeypatch.setattr(auth, "ActivationSession", FakeActivationSession)
    monkeypatch.setattr(auth.db.session, "add", lambda _value: None)
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    monkeypatch.setattr(auth, "_generate_sms_otp", lambda: "123456")
    monkeypatch.setattr(auth, "_send_activation_sms", lambda *_args: True)
    monkeypatch.setattr(
        "security.password_policy.PasswordPolicyService.validate_password",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "services.notifications.activation_email_delivery.try_enqueue_activation_email",
        lambda *_args, **_kwargs: {
            "queued": True,
            "debug_activation_link": "https://example.test/activation",
        },
    )

    response = client.post("/api/v1/auth/register", json=payload)

    assert response.status_code == 201
    assert response.get_json()["activation_email_queued"] is True
    assert created_sessions


@pytest.mark.parametrize(
    ("validated", "result", "status"),
    [
        ({"username": "x"}, None, 400),
        ({"username": "x", "email": "x@example.test"}, None, 400),
        (
            {
                "username": "x",
                "email": "x@example.test",
                "password": "MotDePasse123!",
            },
            SimpleNamespace(
                success=False,
                user=None,
                error={"error": "Cet email existe déjà"},
                status_code=409,
            ),
            409,
        ),
    ],
)
def test_register_branches_metier(client, monkeypatch, validated, result, status):
    monkeypatch.setattr(auth, "validate_request", lambda *_args, **_kwargs: validated)
    monkeypatch.setattr(
        "security.password_policy.PasswordPolicyService.validate_password",
        lambda *_args, **_kwargs: None,
    )
    if result is not None:
        monkeypatch.setattr(
            auth,
            "RegisterUserUseCase",
            lambda: SimpleNamespace(execute=lambda _data: result),
        )
    response = client.post("/api/v1/auth/register", json=validated)
    assert response.status_code == status


def test_verify_email_legacy_succes_deja_confirme_et_expire(client, monkeypatch):
    session = _activation_session()
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(
        auth, "validate_request", lambda *_args, **_kwargs: {"token": "t"}
    )
    monkeypatch.setattr(auth, "_hash_plain_value", lambda _value: "hash")
    monkeypatch.setattr(
        auth,
        "_activation_serializer",
        lambda: SimpleNamespace(
            loads=lambda *_args, **_kwargs: {"sid": "session-test"}
        ),
    )
    monkeypatch.setattr(
        "services.security.activation_legacy.is_legacy_acceptance_active", lambda: True
    )
    monkeypatch.setattr(
        "services.notifications.activation_token.hash_activation_token",
        lambda _token: "modern-hash",
    )
    monkeypatch.setattr(
        "models.activation_email_delivery.ActivationEmailDelivery",
        SimpleNamespace(query=_Query([])),
    )

    success = client.post("/api/v1/auth/activation/verify-email", json={"token": "t"})
    assert success.status_code == 200
    assert session.email_verified_at is not None

    already = client.post("/api/v1/auth/activation/verify-email", json={"token": "t"})
    assert already.status_code == 200

    session.email_verified_at = None
    session.email_token_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    expired = client.post("/api/v1/auth/activation/verify-email", json={"token": "t"})
    assert expired.status_code == 400


def test_activation_sms_finalisation_statut_et_renvois(client, monkeypatch):
    session = _activation_session()
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)

    session.sms_locked_until = datetime.now(UTC) + timedelta(minutes=1)
    locked = client.post(
        "/api/v1/auth/activation/verify-sms",
        json={"activation_session_id": session.activation_session_id, "code": "123456"},
    )
    assert locked.status_code == 429

    session.sms_locked_until = None
    verified = client.post(
        "/api/v1/auth/activation/verify-sms",
        json={"activation_session_id": session.activation_session_id, "code": "123456"},
    )
    assert verified.status_code == 200

    session.email_verified_at = datetime.now(UTC)
    monkeypatch.setattr(auth, "_activation_is_complete", lambda _session: True)
    finalized = client.post(
        "/api/v1/auth/activation/finalize",
        json={"activation_session_id": session.activation_session_id},
    )
    assert finalized.status_code == 200
    assert user.clients[0].is_active is True

    status = client.get(
        "/api/v1/auth/activation/status",
        query_string={"activation_session_id": session.activation_session_id},
    )
    assert status.status_code == 200

    session.consumed_at = None
    session.phone_verified_at = None
    session.last_sms_sent_at = None
    monkeypatch.setattr(auth, "_generate_sms_otp", lambda: "654321")
    monkeypatch.setattr(auth, "_send_activation_sms", lambda *_args: True)
    resent = client.post(
        "/api/v1/auth/activation/resend-sms",
        json={"activation_session_id": session.activation_session_id},
    )
    assert resent.status_code == 200

    session.last_sms_sent_at = None
    updated = client.post(
        "/api/v1/auth/activation/update-phone",
        json={
            "activation_session_id": session.activation_session_id,
            "phone": "+41790000000",
        },
    )
    assert updated.status_code == 200
    assert user.phone == "+41790000000"


def test_activation_renvois_email_branches(client, monkeypatch):
    session = _activation_session()
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    monkeypatch.setattr(
        "services.notifications.activation_email_delivery.can_start_new_delivery_snapshot",
        lambda _session: (True, None),
    )
    monkeypatch.setattr(
        "services.notifications.activation_email_policy.is_same_utc_day",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        "services.notifications.activation_email_policy.enforce_resend_policy",
        lambda **_kwargs: (True, None, 0),
    )
    outcomes = iter(
        [
            {"error": "email_delivery_in_progress"},
            {"error": "cooldown"},
            {"queued": True, "debug_activation_link": "https://example.test/a"},
        ]
    )
    monkeypatch.setattr(
        "services.notifications.activation_email_delivery.try_enqueue_activation_email",
        lambda *_args, **_kwargs: next(outcomes),
    )

    for expected in (429, 429, 200):
        response = client.post(
            "/api/v1/auth/activation/resend-email",
            json={"activation_session_id": session.activation_session_id},
        )
        assert response.status_code == expected


def _patch_mobile_refresh(monkeypatch, sample_user, mobile_session):
    user_dto = SimpleNamespace(public_id=sample_user.public_id)
    monkeypatch.setattr(
        auth,
        "user_repo",
        SimpleNamespace(
            find_by_public_id=lambda _pid: user_dto,
            find_model_by_public_id=lambda _pid: sample_user,
        ),
    )
    monkeypatch.setattr(
        auth, "_validate_refresh_token", lambda _token: ("user-1", None)
    )
    monkeypatch.setattr(auth, "_check_user_profile_active", lambda _user: (True, None))
    monkeypatch.setattr(
        auth,
        "get_jwt",
        lambda: {
            "session_id": mobile_session.session_id,
            "session_epoch": 1,
            "refresh_generation": 1,
        },
    )
    monkeypatch.setattr(
        auth, "get_session_by_id", lambda *_args, **_kwargs: mobile_session
    )
    monkeypatch.setattr(
        auth, "validate_mobile_session", lambda **_kwargs: (None, False)
    )
    monkeypatch.setattr(auth, "resolve_rotation_idempotency", lambda *_a, **_kw: None)
    monkeypatch.setattr(auth, "http_response_for_idempotency", lambda _value: None)
    monkeypatch.setattr(auth, "create_access_token", lambda **_kwargs: "access-new")
    monkeypatch.setattr(auth, "create_refresh_token", lambda **_kwargs: "refresh-new")
    monkeypatch.setattr(auth, "mark_token_rotated", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        auth,
        "store_refresh_token",
        lambda **_kwargs: SimpleNamespace(session_id=None, session_generation=None),
    )
    monkeypatch.setattr(auth, "store_rotation_result", lambda **_kwargs: object())
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    monkeypatch.setattr(auth.db.session, "rollback", lambda: None)
    monkeypatch.setattr(auth.AuditLogger, "log_action", lambda **_kwargs: None)
    monkeypatch.setattr(
        auth,
        "RefreshTokenService",
        lambda: SimpleNamespace(
            touch_token_score=lambda *_args: None,
            revoke_token=lambda *_args: None,
            store_token=lambda *_args, **_kwargs: None,
            limit_active_tokens=lambda *_args: None,
        ),
    )
    monkeypatch.setattr(
        "security.mobile_device_session_service.apply_session_metadata",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "security.mobile_device_session_service.bump_refresh_generation",
        lambda session: setattr(session, "refresh_generation", 2) or 2,
    )


def test_refresh_mobile_rotation_session_succes(client, sample_user, monkeypatch):
    sample_user.public_id = "user-1"
    sample_user.token_version = 1
    mobile_session = SimpleNamespace(
        session_id="session-mobile",
        session_epoch=1,
        refresh_generation=1,
        device_installation_id="device-1",
        is_active=lambda: True,
    )
    _patch_mobile_refresh(monkeypatch, sample_user, mobile_session)

    response = client.post(
        "/api/v1/auth/refresh-token",
        json={"refresh_token": "refresh-old"},
        headers={
            "X-Requested-With": "Expo",
            "X-Device-ID": "device-1",
            "X-Auth-Contract-Version": "mobile-device-session-v1",
            "Idempotency-Key": "refresh-1",
        },
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["access_token"] == "access-new"
    assert body["refresh_generation"] == 2


@pytest.mark.parametrize(
    ("active", "device", "validation_error", "status", "code"),
    [
        (False, "device-1", None, 401, "session_revoked"),
        (True, "autre-device", None, 401, "refresh_replay_detected"),
        (
            True,
            "device-1",
            "session_validation_unavailable",
            503,
            "session_validation_unavailable",
        ),
        (
            True,
            "device-1",
            "session_epoch_mismatch",
            401,
            "rotation_recovery_required",
        ),
    ],
)
def test_refresh_mobile_gardes_session(
    client,
    sample_user,
    monkeypatch,
    active,
    device,
    validation_error,
    status,
    code,
):
    sample_user.public_id = "user-1"
    mobile_session = SimpleNamespace(
        session_id="session-mobile",
        session_epoch=1,
        refresh_generation=2,
        device_installation_id="device-1",
        is_active=lambda: active,
    )
    _patch_mobile_refresh(monkeypatch, sample_user, mobile_session)
    monkeypatch.setattr(
        auth,
        "validate_mobile_session",
        lambda **_kwargs: (validation_error, False),
    )

    response = client.post(
        "/api/v1/auth/refresh-token",
        json={"refresh_token": "refresh-old"},
        headers={
            "X-Requested-With": "Expo",
            "X-Device-ID": device,
            "X-Auth-Contract-Version": "mobile-device-session-v1",
            "Idempotency-Key": "refresh-garde",
        },
    )
    assert response.status_code == status
    assert response.get_json()["error_code"] == code


def test_totp_challenge_succes_echec_et_recuperation(client, sample_user, monkeypatch):
    monkeypatch.setenv("SECURITY_2FA_ENABLED", "true")
    sample_user.totp_secret_encrypted = "secret"
    sample_user.recovery_codes_hash = '["hash"]'
    sample_user.recovery_codes_remaining = 1
    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda _token: {
            "purpose": "2fa_challenge",
            "jti": "jti-test",
            "sub": sample_user.public_id,
        },
    )
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: sample_user)
    monkeypatch.setattr(auth, "User", SimpleNamespace(query=_Query(sample_user)))
    monkeypatch.setattr(
        "security.totp_service.consume_2fa_challenge_jti", lambda _jti: True
    )
    monkeypatch.setattr("security.totp_service.check_2fa_lockout", lambda _uid: False)
    monkeypatch.setattr(
        "security.totp_service.verify_totp_code",
        lambda _secret, code: code == "123456",
    )
    monkeypatch.setattr(
        "security.totp_service.verify_recovery_code",
        lambda _hashes, code: (code == "12345678", "[]"),
    )
    monkeypatch.setattr("security.totp_service.record_2fa_failure", lambda _uid: 1)
    monkeypatch.setattr("security.totp_service.reset_2fa_failures", lambda _uid: None)
    monkeypatch.setattr(
        "security.refresh_token_service.store_refresh_token", lambda **_kwargs: None
    )
    monkeypatch.setattr("shared.audit_helpers.audit_log", lambda *_a, **_kw: None)
    monkeypatch.setattr("services.security.csrf.generate_csrf_token", lambda: "csrf")
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)

    for code, expected in (("000000", 401), ("123456", 200), ("12345678", 200)):
        response = client.post(
            "/api/v1/auth/totp/challenge",
            json={"temp_token": "temp", "code": code},
        )
        assert response.status_code == expected


def test_guest_saferpay_erreurs_statut_et_liaison(
    client, app, sample_user, monkeypatch
):
    payload = {
        "guest_booking_id": "guest-1",
        "status": "paid",
        "promoted_booking_id": 42,
        "public_status_token": "public-token",
        "departure": "A",
        "destination": "B",
    }
    serializer = SimpleNamespace(
        loads=lambda *_args, **_kwargs: {"guest_booking_id": "guest-1"}
    )
    monkeypatch.setattr(auth, "_public_link_serializer", lambda: serializer)
    monkeypatch.setattr(
        auth, "_decode_guest_booking_status_token", lambda _token: ("guest-1", None)
    )
    monkeypatch.setattr(auth, "_public_cache_get", lambda _key: json.dumps(payload))
    monkeypatch.setattr(auth, "_public_cache_setex", lambda *_args: None)
    monkeypatch.setattr("services.saferpay.config.saferpay_configured", lambda: True)
    monkeypatch.setattr(
        "services.guest_saferpay.initialize_guest_saferpay",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("already_promoted")),
    )
    initialized = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/initialize",
        json={"status_token": "token", "guest_booking_id": "guest-1"},
    )
    assert initialized.status_code == 409

    monkeypatch.setattr(
        "services.guest_saferpay.promote_guest_booking_after_saferpay",
        lambda **_kwargs: {"status": "forbidden"},
    )
    asserted = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/assert",
        json={"status_token": "token", "guest_booking_id": "guest-1"},
    )
    assert asserted.status_code == 403

    status = client.get(
        "/api/v1/auth/public/guest-booking/status",
        query_string={"token": "token"},
    )
    assert status.status_code == 200
    assert status.get_json()["status"] == "already_promoted"

    with app.app_context():
        from flask_jwt_extended import create_access_token

        token = create_access_token(
            identity=str(sample_user.public_id),
            additional_claims={"aud": "atmr-api", "role": sample_user.role.value},
        )
    linked = client.post(
        "/api/v1/auth/public/guest-booking/link",
        json={"status_token": "token"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert linked.status_code == 200


def test_activate_account_succes_et_gardes(client, monkeypatch):
    user = _activation_user(
        account_status="invited",
        invite_expires_at=datetime.now(UTC) + timedelta(hours=1),
        invite_token_hash="hash",
        force_password_change=True,
        first_login_completed_at=None,
        authentication_method="email",
        institution_role="member",
        serialize={"public_id": "public-7"},
    )
    user.set_password = lambda password: setattr(user, "password_received", password)
    query = _Query(user)
    monkeypatch.setattr(auth, "User", SimpleNamespace(query=query))
    monkeypatch.setattr(
        "application.institutions.invitation_service.hash_token", lambda _token: "hash"
    )
    monkeypatch.setattr(
        "security.password_policy.PasswordPolicyService.validate_password",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(auth, "create_access_token", lambda **_kwargs: "access")
    monkeypatch.setattr(auth, "create_refresh_token", lambda **_kwargs: "refresh")
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    monkeypatch.setattr(auth.AuditLogger, "log_action", lambda **_kwargs: None)

    success = client.post(
        "/api/v1/auth/activate-account",
        json={"token": "invitation", "password": "MotDePasse123!"},
    )
    assert success.status_code == 200
    assert user.account_status == "active"
    assert user.invite_token_hash is None

    user.account_status = "active"
    user.invite_token_hash = "hash"
    already = client.post(
        "/api/v1/auth/activate-account",
        json={"token": "invitation", "password": "MotDePasse123!"},
    )
    assert already.status_code == 400

    user.account_status = "invited"
    user.invite_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    expired = client.post(
        "/api/v1/auth/activate-account",
        json={"token": "invitation", "password": "MotDePasse123!"},
    )
    assert expired.status_code == 400

    query.value = None
    missing = client.post(
        "/api/v1/auth/activate-account",
        json={"token": "invitation", "password": "MotDePasse123!"},
    )
    assert missing.status_code == 400


def test_logout_mobile_session_et_push_company(client, monkeypatch):
    user = SimpleNamespace(
        id=9,
        public_id="user-9",
        role=UserRole.company,
        company=SimpleNamespace(id=19),
    )
    mobile_session = SimpleNamespace(
        user_id=9,
        session_id="session-9",
        is_active=lambda: True,
    )
    revoked = []
    monkeypatch.setattr(auth, "get_jwt_identity", lambda: user.public_id)
    monkeypatch.setattr(
        auth,
        "get_jwt",
        lambda: {
            "session_id": mobile_session.session_id,
            "jti": "access-jti",
            "exp": datetime.now(UTC).timestamp() + 600,
        },
    )
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: user)
    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda *_args, **_kwargs: {
            "session_id": mobile_session.session_id,
            "sub": user.public_id,
        },
    )
    monkeypatch.setattr(auth, "is_token_revoked", lambda _token: False)
    monkeypatch.setattr(auth, "get_session_by_id", lambda _sid: mobile_session)
    monkeypatch.setattr(
        auth,
        "revoke_mobile_device_session",
        lambda session, **_kwargs: revoked.append(session.session_id),
    )
    monkeypatch.setattr(auth, "revoke_tokens_for_session", lambda *_a, **_kw: None)
    monkeypatch.setattr(auth, "revoke_refresh_token", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        auth,
        "RefreshTokenService",
        lambda: SimpleNamespace(revoke_token=lambda token: revoked.append(token)),
    )
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    monkeypatch.setattr(auth.AuditLogger, "log_action", lambda **_kwargs: None)
    monkeypatch.setattr(
        "security.token_blacklist.revoke_token",
        lambda: True,
    )
    device_token_module = importlib.import_module(
        "application.notifications.upsert_device_token"
    )
    monkeypatch.setattr(
        device_token_module,
        "deactivate_device_tokens_for_logout",
        lambda **_kwargs: 2,
    )
    monkeypatch.setattr(
        "services.security.authentication.AccessTokenService",
        lambda: SimpleNamespace(revoke_token=lambda *_args: None),
    )
    monkeypatch.setattr(
        "services.monitoring.prometheus.track_push_token_invalidated",
        lambda **_kwargs: None,
    )

    response = client.post(
        "/api/v1/auth/logout",
        json={
            "session_id": mobile_session.session_id,
            "refresh_token": "refresh-9",
            "device_id": "device-9",
        },
        headers={"X-Requested-With": "Expo", "X-Device-ID": "device-9"},
    )

    assert response.status_code == 200
    assert mobile_session.session_id in revoked
    assert "refresh-9" in revoked


def test_logout_preuve_invalide_et_legacy(client, monkeypatch):
    user = SimpleNamespace(
        id=9,
        public_id="user-9",
        role=UserRole.client,
        company=None,
    )
    monkeypatch.setattr(auth, "get_jwt_identity", lambda: user.public_id)
    monkeypatch.setattr(auth, "get_jwt", lambda: {})
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: user)
    monkeypatch.setattr(
        auth,
        "RefreshTokenService",
        lambda: SimpleNamespace(revoke_token=lambda _token: None),
    )
    monkeypatch.setattr(auth, "revoke_refresh_token", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "security.token_blacklist.revoke_token",
        lambda: False,
    )
    monkeypatch.setattr(
        "services.security.authentication.AccessTokenService",
        lambda: SimpleNamespace(revoke_token=lambda *_args: None),
    )

    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError()),
    )
    denied = client.post(
        "/api/v1/auth/logout",
        json={"session_id": "inconnue", "refresh_token": "mauvais"},
        headers={"X-Requested-With": "Expo"},
    )
    assert denied.status_code == 401

    legacy = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": "legacy"},
        headers={"X-Requested-With": "Expo"},
    )
    assert legacy.status_code == 200


def test_refresh_erreurs_validation_et_comptes(client, monkeypatch):
    monkeypatch.setattr(auth, "_is_mobile_request", lambda: True)
    monkeypatch.setattr(
        auth,
        "_validate_refresh_token",
        lambda _token: (None, {"error": "store", "_http_status": 503}),
    )
    unavailable = client.post(
        "/api/v1/auth/refresh-token", json={"refresh_token": "token"}
    )
    assert unavailable.status_code == 503

    monkeypatch.setattr(
        auth, "_validate_refresh_token", lambda _token: ("user-1", None)
    )
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: None)
    missing_dto = client.post(
        "/api/v1/auth/refresh-token", json={"refresh_token": "token"}
    )
    assert missing_dto.status_code == 404

    dto = SimpleNamespace(public_id="user-1")
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: dto)
    monkeypatch.setattr(auth.user_repo, "find_model_by_public_id", lambda _pid: None)
    missing_model = client.post(
        "/api/v1/auth/refresh-token", json={"refresh_token": "token"}
    )
    assert missing_model.status_code == 404

    inactive = _activation_user(public_id="user-1")
    monkeypatch.setattr(
        auth.user_repo, "find_model_by_public_id", lambda _pid: inactive
    )
    monkeypatch.setattr(
        auth, "_check_user_profile_active", lambda _user: (False, "Compte suspendu")
    )
    disabled = client.post(
        "/api/v1/auth/refresh-token", json={"refresh_token": "token"}
    )
    assert disabled.status_code == 403


def test_fresh_token_erreurs_et_cookie_web(client, app, sample_user, monkeypatch):
    with app.app_context():
        from flask_jwt_extended import create_access_token

        jwt = create_access_token(
            identity=str(sample_user.public_id),
            additional_claims={"aud": "atmr-api", "role": sample_user.role.value},
        )
    headers = {"Authorization": f"Bearer {jwt}"}
    monkeypatch.setattr(auth, "get_jwt_identity", lambda: None)
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: None)
    assert (
        client.post(
            "/api/v1/auth/fresh-token",
            json={"password": "password123"},
            headers=headers,
        ).status_code
        == 401
    )

    monkeypatch.setattr(auth, "get_jwt_identity", lambda: sample_user.public_id)
    dto = SimpleNamespace(email=sample_user.email)
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: dto)
    monkeypatch.setattr(auth.user_repo, "find_model_by_email", lambda _email: None)
    assert (
        client.post(
            "/api/v1/auth/fresh-token",
            json={"password": "password123"},
            headers=headers,
        ).status_code
        == 401
    )

    monkeypatch.setattr(
        auth.user_repo, "find_model_by_email", lambda _email: sample_user
    )
    missing_password = client.post("/api/v1/auth/fresh-token", json={}, headers=headers)
    assert missing_password.status_code == 400
    success = client.post(
        "/api/v1/auth/fresh-token",
        json={"password": "password123"},
        headers=headers,
    )
    assert success.status_code == 200
    assert "access_token=" in success.headers.get("Set-Cookie", "")


def test_activation_gardes_et_fallbacks(client, monkeypatch):
    session = _activation_session()
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)

    session.consumed_at = datetime.now(UTC)
    finalized = client.post(
        "/api/v1/auth/activation/finalize",
        json={"activation_session_id": session.activation_session_id},
    )
    assert finalized.status_code == 200

    session.consumed_at = None
    session.phone_verified_at = None
    session.last_sms_sent_at = None
    monkeypatch.setattr(auth, "_generate_sms_otp", lambda: "654321")
    monkeypatch.setattr(auth, "_send_activation_sms", lambda *_args: False)
    failed_sms = client.post(
        "/api/v1/auth/activation/resend-sms",
        json={"activation_session_id": session.activation_session_id},
    )
    assert failed_sms.status_code == 502

    session.last_sms_sent_at = None
    failed_update = client.post(
        "/api/v1/auth/activation/update-phone",
        json={
            "activation_session_id": session.activation_session_id,
            "phone": "+41790000001",
        },
    )
    assert failed_update.status_code == 502

    session.consumed_at = datetime.now(UTC)
    conflict = client.post(
        "/api/v1/auth/activation/update-phone",
        json={
            "activation_session_id": session.activation_session_id,
            "phone": "+41790000002",
        },
    )
    assert conflict.status_code == 409


def test_invitation_verification_branches(client, monkeypatch):
    user = _activation_user(
        account_status="invited",
        invite_expires_at=datetime.now(UTC) + timedelta(hours=1),
        institution_id=None,
        institution_role="member",
        first_name="Jean",
        last_name="Test",
    )
    query = _Query(user)
    monkeypatch.setattr(auth, "User", SimpleNamespace(query=query))
    monkeypatch.setattr(
        "application.institutions.invitation_service.hash_token", lambda _token: "hash"
    )

    valid = client.get("/api/v1/auth/invite/token")
    assert valid.status_code == 200

    user.account_status = "active"
    assert client.get("/api/v1/auth/invite/token").status_code == 400

    user.account_status = "invited"
    user.invite_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    assert client.get("/api/v1/auth/invite/token").status_code == 400

    query.value = None
    assert client.get("/api/v1/auth/invite/token").status_code == 400


def test_guest_saferpay_branches_erreur(client, monkeypatch):
    monkeypatch.setattr("services.saferpay.config.saferpay_configured", lambda: True)
    monkeypatch.setattr(
        auth, "_decode_guest_booking_status_token", lambda _token: ("guest-1", None)
    )
    monkeypatch.setattr(auth, "_public_cache_get", lambda _key: "{invalide")
    bad_cache = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/initialize",
        json={"status_token": "token"},
    )
    assert bad_cache.status_code == 404

    monkeypatch.setattr(
        auth,
        "_public_cache_get",
        lambda _key: json.dumps({"guest_booking_id": "guest-1"}),
    )
    monkeypatch.setattr(
        "services.guest_saferpay.initialize_guest_saferpay",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("indisponible")),
    )
    unavailable = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/initialize",
        json={"status_token": "token"},
    )
    assert unavailable.status_code == 503

    monkeypatch.setattr(
        "services.guest_saferpay.promote_guest_booking_after_saferpay",
        lambda **_kwargs: {"status": "payment_failed"},
    )
    failed = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/assert",
        json={"status_token": "token"},
    )
    assert failed.status_code == 200
    assert failed.get_json()["payment_status"] == "failed"


def test_login_mobile_session_durable_complet(app, monkeypatch):
    user = _activation_user(
        public_id="mobile-user",
        username="mobile",
        force_password_change=False,
        password_expires_at=None,
        driver_id=27,
    )
    mobile_session = SimpleNamespace(
        session_id="mobile-session",
        session_epoch=2,
        refresh_generation=3,
        credential_generation=4,
        generation=2,
    )
    auth_result = SimpleNamespace(user=user, error=None)
    monkeypatch.setattr(
        auth,
        "validate_request",
        lambda *_args, **_kwargs: {
            "email": user.email,
            "password": "password123",
            "remember_me": True,
        },
    )
    monkeypatch.setattr(
        auth,
        "AuthenticateUserUseCase",
        lambda: SimpleNamespace(execute=lambda _input: auth_result),
    )
    monkeypatch.setattr(auth, "_check_user_profile_active", lambda _user: (True, None))
    monkeypatch.setattr(auth, "_is_mobile_request", lambda: True)
    monkeypatch.setattr(
        auth,
        "create_or_reuse_session",
        lambda **_kwargs: (
            mobile_session,
            "recovery-credential",
            "revocation-secret",
            ["ancienne-session"],
        ),
    )
    monkeypatch.setattr(auth, "create_access_token", lambda **_kwargs: "access-mobile")
    monkeypatch.setattr(
        auth, "create_refresh_token", lambda **_kwargs: "refresh-mobile"
    )
    monkeypatch.setattr(
        auth,
        "store_refresh_token",
        lambda **_kwargs: SimpleNamespace(session_id=None, session_generation=None),
    )
    monkeypatch.setattr(
        auth,
        "RefreshTokenService",
        lambda: SimpleNamespace(
            store_token=lambda *_args, **_kwargs: None,
            limit_active_tokens=lambda *_args: None,
        ),
    )
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    monkeypatch.setattr(auth.AuditLogger, "log_action", lambda **_kwargs: None)
    monkeypatch.setattr(auth, "_must_complete_onboarding", lambda _user: False)
    monkeypatch.setattr(auth, "_onboarding_reasons", lambda _user: [])
    monkeypatch.setattr(
        "security.mobile_device_session_service.publish_session_revoked",
        lambda _sid: None,
    )

    with app.test_request_context(
        "/api/v1/auth/login",
        method="POST",
        json={"email": user.email, "password": "password123"},
        headers={
            "X-Requested-With": "Expo",
            "X-Device-ID": "device-mobile",
            "X-Auth-Contract-Version": "mobile-device-session-v1",
        },
    ):
        response = auth._login_post_body()

    assert response.status_code == 200
    body = response.get_json()
    assert body["session_id"] == mobile_session.session_id
    assert body["recovery_credential"] == "recovery-credential"
    assert body["revocation_secret"] == "revocation-secret"


def test_login_mobile_gardes_contrat(app, monkeypatch):
    user = _activation_user(
        public_id="mobile-user",
        username="mobile",
        force_password_change=False,
        password_expires_at=None,
    )
    monkeypatch.setattr(
        auth,
        "validate_request",
        lambda *_args, **_kwargs: {
            "email": user.email,
            "password": "password123",
        },
    )
    monkeypatch.setattr(
        auth,
        "AuthenticateUserUseCase",
        lambda: SimpleNamespace(
            execute=lambda _input: SimpleNamespace(user=user, error=None)
        ),
    )
    monkeypatch.setattr(auth, "_check_user_profile_active", lambda _user: (True, None))
    monkeypatch.setattr(auth, "_is_mobile_request", lambda: True)

    with app.test_request_context(
        "/api/v1/auth/login",
        method="POST",
        json={},
        headers={
            "X-Requested-With": "Expo",
            "X-Auth-Contract-Version": "mobile-device-session-v1",
        },
    ):
        missing_device, status = auth._login_post_body()
    assert status == 400
    assert missing_device["error_code"] == "device_identity_required"

    monkeypatch.setattr(
        auth,
        "create_or_reuse_session",
        lambda **_kwargs: (None, None, None, []),
    )
    with app.test_request_context(
        "/api/v1/auth/login",
        method="POST",
        json={},
        headers={
            "X-Requested-With": "Expo",
            "X-Device-ID": "device-mobile",
            "X-Auth-Contract-Version": "mobile-device-session-v1",
        },
    ):
        incomplete, status = auth._login_post_body()
    assert status == 503
    assert incomplete["error_code"] == "mobile_session_contract_incomplete"

    monkeypatch.setattr(
        auth,
        "create_or_reuse_session",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stockage indisponible")),
    )
    with app.test_request_context(
        "/api/v1/auth/login",
        method="POST",
        json={},
        headers={
            "X-Requested-With": "Expo",
            "X-Device-ID": "device-mobile",
            "X-Auth-Contract-Version": "mobile-device-session-v1",
        },
    ):
        failed, status = auth._login_post_body()
    assert status == 503
    assert failed["error_code"] == "session_create_failed"


def test_reinitialisation_mot_de_passe_par_token_branches(
    client, sample_user, monkeypatch
):
    monkeypatch.setattr(auth, "User", SimpleNamespace(query=_Query(sample_user)))
    monkeypatch.setattr(
        auth,
        "_reset_user_password_with_policy",
        lambda _user, _password: ({"message": "ok"}, 200),
    )
    monkeypatch.setattr(
        auth,
        "_activation_serializer",
        lambda: SimpleNamespace(loads=lambda *_args, **_kwargs: sample_user.email),
    )

    missing = client.post("/api/v1/auth/reset-password", json={})
    assert missing.status_code == 400
    success = client.post(
        "/api/v1/auth/reset-password",
        json={"token": "token", "new_password": "MotDePasse123!"},
    )
    assert success.status_code == 200

    monkeypatch.setattr(
        auth,
        "_activation_serializer",
        lambda: SimpleNamespace(
            loads=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                auth.BadSignature("signature invalide")
            )
        ),
    )
    invalid = client.post(
        "/api/v1/auth/reset-password",
        json={"token": "token", "new_password": "MotDePasse123!"},
    )
    assert invalid.status_code == 400

    monkeypatch.setattr(
        auth,
        "_activation_serializer",
        lambda: SimpleNamespace(
            loads=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                auth.SignatureExpired("token expiré")
            )
        ),
    )
    expired = client.post(
        "/api/v1/auth/reset-password",
        json={"token": "token", "new_password": "MotDePasse123!"},
    )
    assert expired.status_code == 400

    monkeypatch.setattr(auth, "User", SimpleNamespace(query=_Query(None)))
    monkeypatch.setattr(
        auth,
        "_activation_serializer",
        lambda: SimpleNamespace(loads=lambda *_args, **_kwargs: sample_user.email),
    )
    unknown = client.post(
        "/api/v1/auth/reset-password",
        json={"token": "token", "new_password": "MotDePasse123!"},
    )
    assert unknown.status_code == 400


def test_envoi_email_activation_succes_et_erreurs(app, monkeypatch):
    user = _activation_user(first_name="Jean")
    monkeypatch.setattr(
        "services.notifications.lirie_email_brand.build_lirie_logo_email_assets",
        lambda: ("cid:logo", []),
    )
    monkeypatch.setattr(auth, "render_template", lambda *_args, **_kwargs: "<p>ok</p>")
    monkeypatch.setattr(auth, "send_email_notification", lambda **_kwargs: {"ok": True})
    with app.app_context():
        auth._send_activation_email(user, "token")

    monkeypatch.setattr(
        auth,
        "send_email_notification",
        lambda **_kwargs: {"ok": False, "error": "fournisseur indisponible"},
    )
    with app.app_context(), pytest.raises(RuntimeError):
        auth._send_activation_email(user, "token")

    user.email = None
    with (
        app.app_context(),
        pytest.raises(ValueError, match="Email utilisateur manquant"),
    ):
        auth._send_activation_email(user, "token")


def test_login_mobile_limite_appareils(app, monkeypatch):
    from security.mobile_device_session_service import DeviceSessionLimitReached

    user = _activation_user(
        public_id="mobile-user",
        username="mobile",
        force_password_change=False,
        password_expires_at=None,
    )
    active_session = SimpleNamespace(serialize=lambda: {"session_id": "ancienne"})
    monkeypatch.setattr(
        auth,
        "validate_request",
        lambda *_args, **_kwargs: {
            "email": user.email,
            "password": "password123",
        },
    )
    monkeypatch.setattr(
        auth,
        "AuthenticateUserUseCase",
        lambda: SimpleNamespace(
            execute=lambda _input: SimpleNamespace(user=user, error=None)
        ),
    )
    monkeypatch.setattr(auth, "_check_user_profile_active", lambda _user: (True, None))
    monkeypatch.setattr(auth, "_is_mobile_request", lambda: True)
    monkeypatch.setattr(
        auth,
        "create_or_reuse_session",
        lambda **_kwargs: (_ for _ in ()).throw(
            DeviceSessionLimitReached([active_session])
        ),
    )
    monkeypatch.setattr(auth.db.session, "rollback", lambda: None)
    monkeypatch.setattr(
        "security.mobile_device_session_service.get_device_session_limit",
        lambda _role: 1,
    )
    monkeypatch.setattr(
        "security.mobile_device_session_service.issue_device_session_resolution_token",
        lambda **_kwargs: "resolution-token",
    )

    with app.test_request_context(
        "/api/v1/auth/login",
        method="POST",
        json={},
        headers={
            "X-Requested-With": "Expo",
            "X-Device-ID": "nouvel-appareil",
            "X-Auth-Contract-Version": "mobile-device-session-v1",
        },
    ):
        body, status = auth._login_post_body()

    assert status == 409
    assert body["resolution_token"] == "resolution-token"
    assert body["sessions"] == [{"session_id": "ancienne"}]


def test_forgot_password_fournisseur_indisponible(client, sample_user, monkeypatch):
    monkeypatch.setattr(auth.user_repo, "find_by_email", lambda _email: sample_user)
    monkeypatch.setattr(
        auth,
        "send_email_notification",
        lambda **_kwargs: {"ok": False, "error": "smtp indisponible"},
    )
    response = client.post(
        "/api/v1/auth/forgot-password",
        json={"email": sample_user.email},
    )
    assert response.status_code == 200
    assert response.get_json()["reason"] == "forgot_password_email_unavailable"


def test_revoquer_session_specifique(client, app, sample_user, monkeypatch):
    session = SimpleNamespace(
        user_id=sample_user.id,
        is_revoked=False,
        revoked_at=None,
        revoked_reason=None,
    )
    refresh_module = importlib.import_module("models.refresh_token")
    monkeypatch.setattr(
        refresh_module,
        "RefreshToken",
        SimpleNamespace(query=_Query(session)),
    )
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: sample_user)
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    monkeypatch.setattr("shared.audit_helpers.audit_log", lambda *_a, **_kw: None)
    with app.app_context():
        from flask_jwt_extended import create_access_token

        token = create_access_token(
            identity=str(sample_user.public_id),
            additional_claims={"aud": "atmr-api", "role": sample_user.role.value},
        )
    response = client.delete(
        "/api/v1/auth/sessions/42",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 204
    assert session.is_revoked is True

    already = client.delete(
        "/api/v1/auth/sessions/42",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert already.status_code == 409


def test_login_exception_et_origine_refusee(client, monkeypatch):
    monkeypatch.setattr(auth, "_is_mobile_request", lambda: False)
    monkeypatch.setattr(
        auth, "validate_login_origin_for_web", lambda: (False, "origin_refused")
    )
    refused = client.post(
        "/api/v1/auth/login",
        json={"email": "test@example.test", "password": "password123"},
    )
    assert refused.status_code == 403

    monkeypatch.setattr(auth, "_is_mobile_request", lambda: True)
    monkeypatch.setattr(
        auth,
        "_login_post_body",
        lambda: (_ for _ in ()).throw(RuntimeError("erreur interne")),
    )
    monkeypatch.setattr(auth.AuditLogger, "log_action", lambda **_kwargs: None)
    failed = client.post(
        "/api/v1/auth/login",
        json={"email": "test@example.test", "password": "password123"},
        headers={"X-Requested-With": "Expo"},
    )
    assert failed.status_code == 500


def test_profils_institution_et_limites_refresh(monkeypatch):
    monkeypatch.setattr(
        auth, "enforce_demo_user_access_validity", lambda _user: (True, None)
    )
    institution = SimpleNamespace(
        role=UserRole.institution,
        institution_id=1,
        account_status="active",
        archived_at=datetime.now(UTC),
        force_password_change=False,
        password_expires_at=None,
    )
    assert auth._check_user_profile_active(institution)[0] is False
    institution.archived_at = None
    institution.account_status = "disabled"
    assert auth._check_user_profile_active(institution)[0] is False
    institution.account_status = "invited"
    assert auth._check_user_profile_active(institution)[0] is False
    institution.account_status = "active"
    institution.force_password_change = True
    institution.password_expires_at = datetime.now() - timedelta(days=1)
    assert auth._check_user_profile_active(institution)[0] is False

    assert (
        auth._resolve_max_active_refresh_tokens(SimpleNamespace(role=UserRole.driver))
        == 15
    )
    assert auth._resolve_max_active_refresh_tokens(institution) == 15
    assert auth._get_password_hash_version(SimpleNamespace(password="")) == ""


def test_validation_refresh_fail_closed(app, sample_user, monkeypatch):
    dto = SimpleNamespace(public_id=sample_user.public_id)
    repo = SimpleNamespace(
        find_by_public_id=lambda _pid: dto,
        find_model_by_public_id=lambda _pid: sample_user,
    )
    monkeypatch.setattr(auth, "user_repo", repo)
    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda *_args, **_kwargs: {
            "sub": sample_user.public_id,
            "type": "refresh",
            "pwd_hash": "ancien",
        },
    )
    monkeypatch.setattr(auth, "is_token_revoked", lambda *_a, **_kw: False)
    monkeypatch.setattr(auth, "refresh_fail_closed_enabled", lambda: True)
    monkeypatch.setattr(auth, "_get_password_hash_version", lambda _user: "nouveau")
    with app.test_request_context("/"):
        assert auth._validate_refresh_token("token")[0] is None

    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda *_args, **_kwargs: {
            "sub": sample_user.public_id,
            "type": "refresh",
        },
    )
    monkeypatch.setattr(
        auth,
        "RefreshTokenService",
        lambda: SimpleNamespace(is_token_valid=lambda *_a, **_kw: False),
    )
    with app.test_request_context("/"):
        assert auth._validate_refresh_token("token")[0] is None

    monkeypatch.setattr(
        auth,
        "RefreshTokenService",
        lambda: SimpleNamespace(
            is_token_valid=lambda *_a, **_kw: (_ for _ in ()).throw(
                auth.RefreshStoreUnavailableError()
            )
        ),
    )
    with app.test_request_context("/"):
        error = auth._validate_refresh_token("token")[1]
        assert error["_http_status"] == 503

    monkeypatch.setattr(
        auth,
        "RefreshTokenService",
        lambda: SimpleNamespace(is_token_valid=lambda *_a, **_kw: True),
    )
    monkeypatch.setattr(
        auth,
        "is_token_revoked",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("redis")),
    )
    with app.test_request_context("/"):
        error = auth._validate_refresh_token("token")[1]
        assert error["_http_status"] == 503


def test_verify_email_modern_branches(client, monkeypatch):
    session = _activation_session(email_delivery_id="delivery-1")
    user = _activation_user()
    delivery = SimpleNamespace(
        email_token_hash="hash",
        email_delivery_id="delivery-1",
        token_key_version=1,
        activation_session_pk=session.id,
        superseded_at=None,
        token_expires_at=datetime.now(UTC) + timedelta(minutes=5),
    )
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda *_a, **_kw: {"token": "token"})
    monkeypatch.setattr(
        "services.notifications.activation_token.hash_activation_token",
        lambda _token: "hash",
    )
    monkeypatch.setattr(
        "services.notifications.activation_email_delivery.get_activation_session_for_update",
        lambda _pk: session,
    )
    monkeypatch.setattr(
        "models.activation_email_delivery.ActivationEmailDelivery",
        SimpleNamespace(query=_Query([delivery])),
    )

    monkeypatch.setattr(
        "services.notifications.activation_token.verify_activation_token",
        lambda *_a, **_kw: False,
    )
    invalid = client.post(
        "/api/v1/auth/activation/verify-email", json={"token": "token"}
    )
    assert invalid.status_code == 400

    monkeypatch.setattr(
        "services.notifications.activation_token.verify_activation_token",
        lambda *_a, **_kw: True,
    )
    delivery.superseded_at = datetime.now(UTC)
    superseded = client.post(
        "/api/v1/auth/activation/verify-email", json={"token": "token"}
    )
    assert superseded.status_code == 400

    delivery.superseded_at = None
    delivery.token_expires_at = None
    invalid_expiry = client.post(
        "/api/v1/auth/activation/verify-email", json={"token": "token"}
    )
    assert invalid_expiry.status_code == 400

    delivery.token_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    delivery.activation_session_pk = 999
    mismatched = client.post(
        "/api/v1/auth/activation/verify-email", json={"token": "token"}
    )
    assert mismatched.status_code == 400

    delivery.activation_session_pk = session.id
    session.email_verified_at = datetime.now(UTC)
    already_verified = client.post(
        "/api/v1/auth/activation/verify-email", json={"token": "token"}
    )
    assert already_verified.status_code == 200


def test_changement_mot_de_passe_branches(client, app, sample_user, monkeypatch):
    monkeypatch.setattr(auth, "User", SimpleNamespace(query=_Query(sample_user)))
    with app.app_context():
        from flask_jwt_extended import create_access_token

        token = create_access_token(
            identity=str(sample_user.public_id),
            additional_claims={"aud": "atmr-api", "role": sample_user.role.value},
        )
    headers = {"Authorization": f"Bearer {token}"}
    missing = client.post("/api/v1/auth/change-password", json={}, headers=headers)
    assert missing.status_code == 400
    mismatch = client.post(
        "/api/v1/auth/change-password",
        json={"new_password": "Nouveau123!", "confirm_password": "Autre123!"},
        headers=headers,
    )
    assert mismatch.status_code == 400
    sample_user.force_password_change = False
    denied = client.post(
        "/api/v1/auth/change-password",
        json={
            "new_password": "Nouveau123!",
            "confirm_password": "Nouveau123!",
            "current_password": "incorrect",
        },
        headers=headers,
    )
    assert denied.status_code == 401
    sample_user.force_password_change = True
    monkeypatch.setattr(
        auth,
        "_reset_user_password_with_policy",
        lambda _user, _password: ({"message": "ok"}, 200),
    )
    success = client.post(
        "/api/v1/auth/change-password",
        json={"new_password": "Nouveau123!", "confirm_password": "Nouveau123!"},
        headers=headers,
    )
    assert success.status_code == 200


def test_totp_challenge_gardes_supplementaires(client, sample_user, monkeypatch):
    monkeypatch.setenv("SECURITY_2FA_ENABLED", "true")
    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda _token: {
            "purpose": "2fa_challenge",
            "jti": "jti",
            "sub": sample_user.public_id,
        },
    )
    monkeypatch.setattr(
        "security.totp_service.consume_2fa_challenge_jti", lambda _jti: False
    )
    used = client.post(
        "/api/v1/auth/totp/challenge",
        json={"temp_token": "temp", "code": "123456"},
    )
    assert used.status_code == 401

    monkeypatch.setattr(
        "security.totp_service.consume_2fa_challenge_jti", lambda _jti: True
    )
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: None)
    missing = client.post(
        "/api/v1/auth/totp/challenge",
        json={"temp_token": "temp", "code": "123456"},
    )
    assert missing.status_code == 404

    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: sample_user)
    monkeypatch.setattr("security.totp_service.check_2fa_lockout", lambda _uid: True)
    locked = client.post(
        "/api/v1/auth/totp/challenge",
        json={"temp_token": "temp", "code": "123456"},
    )
    assert locked.status_code == 429


@pytest.mark.parametrize(
    ("url", "payload"),
    [
        ("/api/v1/auth/activation/verify-email", {"token": "token"}),
        (
            "/api/v1/auth/activation/verify-sms",
            {"activation_session_id": "session", "code": "123456"},
        ),
        (
            "/api/v1/auth/activation/finalize",
            {"activation_session_id": "session"},
        ),
        (
            "/api/v1/auth/activation/resend-email",
            {"activation_session_id": "session"},
        ),
        (
            "/api/v1/auth/activation/resend-sms",
            {"activation_session_id": "session"},
        ),
        (
            "/api/v1/auth/activation/update-phone",
            {"activation_session_id": "session", "phone": "+41791234567"},
        ),
    ],
)
def test_activation_gestion_exceptions(client, monkeypatch, url, payload):
    monkeypatch.setattr(
        auth,
        "validate_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("test")),
    )
    monkeypatch.setattr(auth.sentry_sdk, "capture_exception", lambda _error: None)
    monkeypatch.setattr(
        auth.APIErrorHandler,
        "handle_exception",
        lambda _error, _logger: ({"error": "interne"}, 500),
    )
    monkeypatch.setattr(auth.db.session, "rollback", lambda: None)
    response = client.post(url, json=payload)
    assert response.status_code == 500


def test_activation_erreurs_validation(client, monkeypatch):
    monkeypatch.setattr(
        auth,
        "validate_request",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            auth.ValidationError({"champ": ["invalide"]})
        ),
    )
    cases = [
        ("/api/v1/auth/activation/verify-email", {"token": "token"}),
        (
            "/api/v1/auth/activation/verify-sms",
            {"activation_session_id": "session", "code": "123456"},
        ),
        (
            "/api/v1/auth/activation/finalize",
            {"activation_session_id": "session"},
        ),
        (
            "/api/v1/auth/activation/resend-email",
            {"activation_session_id": "session"},
        ),
        (
            "/api/v1/auth/activation/resend-sms",
            {"activation_session_id": "session"},
        ),
        (
            "/api/v1/auth/activation/update-phone",
            {"activation_session_id": "session", "phone": "+41791234567"},
        ),
    ]
    for url, payload in cases:
        assert client.post(url, json=payload).status_code == 400


def test_csrf_identites_dict_et_nombre(client, monkeypatch):
    monkeypatch.setattr(auth, "generate_csrf_token", lambda **_kwargs: "csrf-test")
    monkeypatch.setattr(auth, "get_jwt_identity", lambda: 42)
    numeric = client.get("/api/v1/auth/csrf-token")
    assert numeric.status_code == 200
    monkeypatch.setattr(auth, "get_jwt_identity", lambda: {"user_id": 43})
    mapping = client.get("/api/v1/auth/csrf-token")
    assert mapping.status_code == 200


def test_passwordless_codes_invalides_et_limite(client, app, monkeypatch):
    monkeypatch.setitem(app.config, "ENVIRONMENT", "development")
    payload = {
        "otp_session_id": "otp-test",
        "user_public_id": "user",
        "code_hash": auth._hash_plain_value("123456"),
        "attempts": 5,
        "max_attempts": 5,
    }
    monkeypatch.setattr(auth, "_public_cache_get", lambda _key: json.dumps(payload))
    monkeypatch.setattr(auth, "_public_cache_setex", lambda *_args: None)
    limited = client.post(
        "/api/v1/auth/passwordless/otp/verify",
        json={"otp_session_id": "otp-test", "code": "000000"},
    )
    assert limited.status_code == 429

    payload["attempts"] = 0
    invalid = client.post(
        "/api/v1/auth/passwordless/otp/verify",
        json={"otp_session_id": "otp-test", "code": "000000"},
    )
    assert invalid.status_code == 401


def test_activation_sessions_absentes(client, monkeypatch):
    monkeypatch.setattr(auth, "ActivationSession", SimpleNamespace(query=_Query(None)))
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    cases = [
        (
            "/api/v1/auth/activation/verify-sms",
            {"activation_session_id": "absente", "code": "123456"},
        ),
        (
            "/api/v1/auth/activation/finalize",
            {"activation_session_id": "absente"},
        ),
        (
            "/api/v1/auth/activation/resend-email",
            {"activation_session_id": "absente"},
        ),
        (
            "/api/v1/auth/activation/resend-sms",
            {"activation_session_id": "absente"},
        ),
        (
            "/api/v1/auth/activation/update-phone",
            {"activation_session_id": "absente", "phone": "+41791234567"},
        ),
    ]
    for url, payload in cases:
        assert client.post(url, json=payload).status_code == 404


@pytest.mark.parametrize("role", [UserRole.driver, UserRole.company])
def test_switch_context_refus_interprofils(client, app, sample_user, monkeypatch, role):
    sample_user.role = role
    contexts = [
        {
            "context_id": "company:1",
            "context_type": "company",
            "is_default": True,
            "allow_mobile_context_switch": False,
        },
        {
            "context_id": "driver:2",
            "context_type": "driver",
            "allow_mobile_context_switch": False,
        },
    ]
    monkeypatch.setattr(auth, "_load_user_for_bootstrap", lambda _pid: sample_user)
    monkeypatch.setattr(auth, "_prepare_user_for_bootstrap", lambda user: user)
    monkeypatch.setattr(auth, "_build_available_contexts", lambda _user: contexts)
    monkeypatch.setattr(auth, "_get_saved_active_context", lambda _pid: "company:1")
    with app.app_context():
        from flask_jwt_extended import create_access_token

        token = create_access_token(
            identity=str(sample_user.public_id),
            additional_claims={"aud": "atmr-api", "role": role.value},
        )
    response = client.post(
        "/api/v1/auth/switch-context",
        json={"target_context_id": "driver:2"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 403


def test_login_json_malforme_et_sms_sans_telephone(app):
    with app.test_request_context(
        "/api/v1/auth/login",
        method="POST",
        data="{",
        content_type="application/json",
    ):
        response = auth._login_post_body()
    assert response[1] == 400
    assert auth._send_activation_sms(SimpleNamespace(phone=None), "123456") is False


def test_verify_email_legacy_signatures_et_doublon(client, monkeypatch):
    delivery = SimpleNamespace(email_token_hash="hash")
    monkeypatch.setattr(auth, "validate_request", lambda *_a, **_kw: {"token": "token"})
    monkeypatch.setattr(
        "services.notifications.activation_token.hash_activation_token",
        lambda _token: "hash",
    )
    delivery_model = SimpleNamespace(query=_Query([delivery, delivery]))
    monkeypatch.setattr(
        "models.activation_email_delivery.ActivationEmailDelivery", delivery_model
    )
    duplicate = client.post(
        "/api/v1/auth/activation/verify-email", json={"token": "token"}
    )
    assert duplicate.status_code == 400

    delivery_model.query = _Query([])
    monkeypatch.setattr(
        "services.security.activation_legacy.is_legacy_acceptance_active", lambda: True
    )
    exceptions = iter(
        [
            auth.SignatureExpired("expirée"),
            auth.BadSignature("invalide"),
        ]
    )
    monkeypatch.setattr(
        auth,
        "_activation_serializer",
        lambda: SimpleNamespace(
            loads=lambda *_args, **_kwargs: (_ for _ in ()).throw(next(exceptions))
        ),
    )
    expired = client.post(
        "/api/v1/auth/activation/verify-email", json={"token": "token"}
    )
    invalid = client.post(
        "/api/v1/auth/activation/verify-email", json={"token": "token"}
    )
    assert expired.status_code == 400
    assert invalid.status_code == 400


def test_activation_branches_idempotentes_et_envoi_occupe(client, monkeypatch):
    session = _activation_session(phone_verified_at=datetime.now(UTC))
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)

    verified = client.post(
        "/api/v1/auth/activation/verify-sms",
        json={"activation_session_id": session.activation_session_id, "code": "123456"},
    )
    assert verified.status_code == 200

    session.email_verified_at = datetime.now(UTC)
    already_email = client.post(
        "/api/v1/auth/activation/resend-email",
        json={"activation_session_id": session.activation_session_id},
    )
    assert already_email.status_code == 200

    session.email_verified_at = None
    monkeypatch.setattr(
        "services.notifications.activation_email_delivery.can_start_new_delivery_snapshot",
        lambda _session: (False, "sending"),
    )
    busy = client.post(
        "/api/v1/auth/activation/resend-email",
        json={"activation_session_id": session.activation_session_id},
    )
    assert busy.status_code == 429

    monkeypatch.setattr(auth, "_activation_is_complete", lambda _session: True)
    monkeypatch.setattr(auth, "User", SimpleNamespace(query=_Query(None)))
    session.phone_verified_at = datetime.now(UTC)
    missing_user = client.post(
        "/api/v1/auth/activation/finalize",
        json={"activation_session_id": session.activation_session_id},
    )
    assert missing_user.status_code == 404


def test_finalisation_email_manquant_et_totp_sans_identite(client, monkeypatch):
    session = _activation_session(phone_verified_at=datetime.now(UTC))
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    monkeypatch.setattr(auth, "_activation_is_complete", lambda _session: False)
    monkeypatch.setattr(
        auth, "_activation_channel_requirements", lambda _session: (True, False)
    )
    incomplete = client.post(
        "/api/v1/auth/activation/finalize",
        json={"activation_session_id": session.activation_session_id},
    )
    assert incomplete.status_code == 400

    monkeypatch.setenv("SECURITY_2FA_ENABLED", "true")
    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda _token: {"purpose": "2fa_challenge", "jti": None, "sub": None},
    )
    invalid = client.post(
        "/api/v1/auth/totp/challenge",
        json={"temp_token": "temp", "code": "123456"},
    )
    assert invalid.status_code == 401


def test_refresh_collisions_idempotence_et_stockage(
    client, app, sample_user, monkeypatch
):
    """Les collisions de rotation sont récupérées, sinon fermées explicitement."""
    sample_user.public_id = "user-1"
    sample_user.token_version = 1
    mobile_session = SimpleNamespace(
        session_id="session-mobile",
        session_epoch=1,
        refresh_generation=1,
        device_installation_id="device-1",
        is_active=lambda: True,
    )
    _patch_mobile_refresh(monkeypatch, sample_user, mobile_session)
    headers = {
        "X-Requested-With": "Expo",
        "X-Device-ID": "device-1",
        "X-Auth-Contract-Version": "mobile-device-session-v1",
        "Idempotency-Key": "collision-1",
    }

    monkeypatch.setattr(auth, "store_rotation_result", lambda **_kwargs: None)
    monkeypatch.setattr(
        auth, "resolve_rotation_idempotency", lambda *_args, **_kwargs: {"winner": True}
    )
    monkeypatch.setattr(
        auth,
        "http_response_for_idempotency",
        lambda value: ({"recovered": True}, 200) if value else None,
    )
    recovered = client.post(
        "/api/v1/auth/refresh-token",
        json={"refresh_token": "refresh-old"},
        headers=headers,
    )
    assert recovered.status_code == 200
    assert recovered.get_json()["recovered"] is True

    monkeypatch.setattr(auth, "store_rotation_result", lambda **_kwargs: object())
    monkeypatch.setattr(auth, "resolve_rotation_idempotency", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        auth,
        "mark_token_rotated",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("collision")),
    )
    monkeypatch.setattr(auth, "is_rotation_idempotency_conflict", lambda _error: True)
    recovered_exception = client.post(
        "/api/v1/auth/refresh-token",
        json={"refresh_token": "refresh-old"},
        headers={**headers, "Idempotency-Key": "collision-2"},
    )
    assert recovered_exception.status_code == 401

    monkeypatch.setattr(auth, "is_rotation_idempotency_conflict", lambda _error: False)
    monkeypatch.setattr(auth, "refresh_fail_closed_enabled", lambda: True)
    monkeypatch.setattr(auth, "resolve_rotation_idempotency", lambda *_a, **_kw: None)
    monkeypatch.setitem(app.config, "TESTING", False)
    unavailable = client.post(
        "/api/v1/auth/refresh-token",
        json={"refresh_token": "refresh-old"},
        headers={**headers, "Idempotency-Key": "collision-3"},
    )
    assert unavailable.status_code == 503


def test_refresh_conflit_sans_gagnant_et_exception_globale(
    client, sample_user, monkeypatch
):
    sample_user.public_id = "user-1"
    mobile_session = SimpleNamespace(
        session_id="session-mobile",
        session_epoch=1,
        refresh_generation=1,
        device_installation_id="device-1",
        is_active=lambda: True,
    )
    _patch_mobile_refresh(monkeypatch, sample_user, mobile_session)
    monkeypatch.setattr(auth, "store_rotation_result", lambda **_kwargs: None)
    monkeypatch.setattr(auth, "resolve_rotation_idempotency", lambda *_a, **_kw: None)
    headers = {
        "X-Requested-With": "Expo",
        "X-Device-ID": "device-1",
        "X-Auth-Contract-Version": "mobile-device-session-v1",
        "Idempotency-Key": "sans-gagnant",
    }
    conflict = client.post(
        "/api/v1/auth/refresh-token",
        json={"refresh_token": "refresh-old"},
        headers=headers,
    )
    assert conflict.status_code == 401
    assert conflict.get_json()["error_code"] == "rotation_recovery_required"

    monkeypatch.setattr(
        auth,
        "_validate_refresh_token",
        lambda _token: (_ for _ in ()).throw(RuntimeError("inattendue")),
    )
    monkeypatch.setattr(
        auth.APIErrorHandler,
        "handle_exception",
        lambda _error, _logger: ({"error": "interne"}, 500),
    )
    failed = client.post(
        "/api/v1/auth/refresh-token",
        json={"refresh_token": "refresh-old"},
        headers=headers,
    )
    assert failed.status_code == 500


def _patch_logout_common(monkeypatch, user, claims):
    monkeypatch.setattr(auth, "get_jwt_identity", lambda: user.public_id)
    monkeypatch.setattr(auth, "get_jwt", lambda: claims)
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: user)
    monkeypatch.setattr(
        auth,
        "RefreshTokenService",
        lambda: SimpleNamespace(revoke_token=lambda _token: None),
    )
    monkeypatch.setattr(auth, "revoke_refresh_token", lambda *_a, **_kw: None)
    monkeypatch.setattr(auth, "revoke_tokens_for_session", lambda *_a, **_kw: None)
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    monkeypatch.setattr(
        "security.token_blacklist.revoke_token",
        lambda: True,
    )
    monkeypatch.setattr(
        "services.security.authentication.AccessTokenService",
        lambda: SimpleNamespace(revoke_token=lambda *_args: None),
    )


def test_logout_web_preuves_session_et_idempotence(client, monkeypatch):
    user = SimpleNamespace(
        id=9,
        public_id="user-9",
        role=UserRole.client,
        company=None,
    )
    _patch_logout_common(monkeypatch, user, {})

    denied = client.post("/api/v1/auth/logout", json={"session_id": "session-9"})
    assert denied.status_code == 401
    assert "access_token=;" in denied.headers.get("Set-Cookie", "")

    monkeypatch.setattr(
        auth,
        "get_jwt",
        lambda: {"session_id": "session-9", "jti": "jti", "exp": 0},
    )
    foreign = SimpleNamespace(
        user_id=999,
        session_id="session-9",
        is_active=lambda: True,
    )
    monkeypatch.setattr(auth, "get_session_by_id", lambda _sid: foreign)
    mismatch = client.post("/api/v1/auth/logout", json={"session_id": "session-9"})
    assert mismatch.status_code == 401

    terminal = SimpleNamespace(
        user_id=user.id,
        session_id="session-9",
        is_active=lambda: False,
    )
    monkeypatch.setattr(auth, "get_session_by_id", lambda _sid: terminal)
    already = client.post("/api/v1/auth/logout", json={"session_id": "session-9"})
    assert already.status_code == 200
    assert already.get_json()["already_revoked"] is True

    monkeypatch.setattr(auth, "get_session_by_id", lambda _sid: None)
    absent = client.post("/api/v1/auth/logout", json={"session_id": "session-9"})
    assert absent.status_code == 200
    assert absent.get_json()["already_revoked"] is True


def test_logout_driver_legacy_push_et_erreurs(client, monkeypatch):
    user = SimpleNamespace(
        id=9,
        public_id="user-9",
        role=UserRole.driver,
        company=None,
    )
    _patch_logout_common(monkeypatch, user, {})
    driver = SimpleNamespace(id=27)
    monkeypatch.setattr(
        "repositories.driver_repository.DriverRepository",
        lambda: SimpleNamespace(find_model_by_user_id=lambda _uid: driver),
    )
    monkeypatch.setattr(
        "security.refresh_token_service.revoke_active_tokens_for_device",
        lambda *_args, **_kwargs: None,
    )
    device_token_module = importlib.import_module(
        "application.notifications.upsert_device_token"
    )
    monkeypatch.setattr(
        device_token_module,
        "deactivate_device_tokens_for_logout",
        lambda **_kwargs: 0,
    )
    no_push = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": "legacy", "device_id": "device-9"},
        headers={"X-Requested-With": "Expo"},
    )
    assert no_push.status_code == 200

    no_device = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": "legacy"},
        headers={"X-Requested-With": "Expo"},
    )
    assert no_device.status_code == 200

    monkeypatch.setattr(
        "repositories.driver_repository.DriverRepository",
        lambda: (_ for _ in ()).throw(RuntimeError("driver indisponible")),
    )
    ignored = client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": "legacy", "device_id": "device-9"},
        headers={"X-Requested-With": "Expo"},
    )
    assert ignored.status_code == 200


def test_logout_erreurs_revoke_et_fallback_web(client, monkeypatch):
    user = SimpleNamespace(
        id=9,
        public_id="user-9",
        role=UserRole.client,
        company=None,
    )
    _patch_logout_common(
        monkeypatch,
        user,
        {
            "session_id": "session-9",
            "jti": "jti",
            "exp": datetime.now(UTC).timestamp() + 60,
        },
    )
    session = SimpleNamespace(
        user_id=user.id,
        session_id="session-9",
        is_active=lambda: True,
    )
    monkeypatch.setattr(auth, "get_session_by_id", lambda _sid: session)
    monkeypatch.setattr(
        auth,
        "revoke_mobile_device_session",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("révocation")),
    )
    monkeypatch.setattr(
        "services.security.authentication.AccessTokenService",
        lambda: SimpleNamespace(
            revoke_token=lambda *_args: (_ for _ in ()).throw(RuntimeError("access"))
        ),
    )
    monkeypatch.setattr(
        auth.AuditLogger,
        "log_action",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("audit")),
    )
    response = client.post("/api/v1/auth/logout", json={"session_id": "session-9"})
    assert response.status_code == 200

    monkeypatch.setattr(
        auth,
        "get_jwt_identity",
        lambda: (_ for _ in ()).throw(RuntimeError("jwt")),
    )
    fallback = client.post("/api/v1/auth/logout", json={})
    assert fallback.status_code == 200
    assert fallback.get_json()["warning"] == "partial"


def test_register_options_validation_et_resultats(client, app, monkeypatch):
    monkeypatch.setattr(
        auth,
        "validate_request",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            auth.ValidationError({"email": ["invalide"]})
        ),
    )
    invalid = client.post("/api/v1/auth/register", json={})
    assert invalid.status_code == 400

    validated = {
        "username": "nouveau",
        "email": "nouveau@example.test",
        "password": "MotDePasse123!",
    }
    monkeypatch.setattr(auth, "validate_request", lambda *_a, **_kw: validated)
    monkeypatch.setattr(
        "security.password_policy.PasswordPolicyService.validate_password",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        auth,
        "RegisterUserUseCase",
        lambda: SimpleNamespace(
            execute=lambda _data: SimpleNamespace(
                success=False,
                user=None,
                error={"error": "Service temporairement indisponible"},
                status_code=422,
            )
        ),
    )
    failed = client.post("/api/v1/auth/register", json=validated)
    assert failed.status_code == 422

    with app.test_request_context("/api/v1/auth/register", method="OPTIONS"):
        options = auth.Register().post()
    assert options[1] == 204


def test_register_sms_exception_et_echec_email(client, monkeypatch):
    payload = {
        "username": "nouveau",
        "email": "nouveau@example.test",
        "password": "MotDePasse123!",
        "phone": "+41791234567",
    }
    user = _activation_user(username="nouveau", email=payload["email"])
    result = SimpleNamespace(success=True, user=user, error=None, status_code=201)

    class FakeActivationSession:
        activation_session_id = "session-register"

    monkeypatch.setattr(auth, "validate_request", lambda *_a, **_kw: payload)
    monkeypatch.setattr(
        auth,
        "RegisterUserUseCase",
        lambda: SimpleNamespace(execute=lambda _data: result),
    )
    monkeypatch.setattr(auth, "Client", type("FakeClient", (), {"id": 17}))
    monkeypatch.setattr(auth, "ActivationSession", FakeActivationSession)
    monkeypatch.setattr(auth.db.session, "add", lambda _value: None)
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    monkeypatch.setattr(auth, "_generate_sms_otp", lambda: "123456")
    monkeypatch.setattr(
        auth,
        "_send_activation_sms",
        lambda *_a: (_ for _ in ()).throw(RuntimeError("sms")),
    )
    monkeypatch.setattr(
        "security.password_policy.PasswordPolicyService.validate_password",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "services.notifications.activation_email_delivery.try_enqueue_activation_email",
        lambda *_a, **_kw: {
            "require_502": True,
            "debug_activation_link": "https://example.test/debug",
        },
    )
    failed = client.post("/api/v1/auth/register", json=payload)
    assert failed.status_code == 502
    assert failed.get_json()["debug_activation_link"]


def test_contextes_helpers_et_configuration(app, monkeypatch):
    company = SimpleNamespace(id=3, name="Transport")
    driver = SimpleNamespace(id=4, company_id=3, company=company)
    base = {
        "public_id": "public",
        "clients": [SimpleNamespace()],
        "driver": driver,
        "company": company,
        "institution_id": 5,
    }
    for role in (
        UserRole.client,
        UserRole.driver,
        UserRole.company,
        UserRole.institution,
        UserRole.admin,
    ):
        contexts = auth._build_available_contexts(SimpleNamespace(role=role, **base))
        assert contexts

    assert auth._company_allows_driver_workspace_switch(None) is False
    assert (
        auth._allow_mobile_company_driver_context_switch(
            SimpleNamespace(role=UserRole.company),
            context_type="client",
            company=company,
        )
        is False
    )
    assert auth._resolve_company_id(SimpleNamespace(role=UserRole.admin)) is None

    with app.app_context():
        monkeypatch.setitem(app.config, "MOBILE_FEATURE_FLAGS", "{invalide")
        monkeypatch.setattr(
            "services.saferpay.config.saferpay_configured",
            lambda: (_ for _ in ()).throw(RuntimeError("saferpay")),
        )
        monkeypatch.setattr(
            "services.infrastructure.runtime_flags.get_mobile_startup_runtime_flags",
            lambda: (_ for _ in ()).throw(RuntimeError("flags")),
        )
        flags = auth._feature_flags_config()
        assert flags["saferpay_enabled"] is False
        assert flags["ios_startup_fatal_recovery_disabled"] is False

    redis = SimpleNamespace(
        get=lambda _key: b"driver:4",
        setex=lambda *_args: None,
    )
    monkeypatch.setattr(auth, "redis_client", redis)
    assert auth._get_saved_active_context("public") == "driver:4"
    auth._save_active_context("public", "company:3")
    monkeypatch.setattr(auth, "redis_client", None)
    assert auth._get_saved_active_context("public") is None
    auth._save_active_context("public", "company:3")


def test_branches_publiques_restantes(client, app, monkeypatch):
    service_payload = {
        "departure": "Hopital Lausanne",
        "destination": "Pully",
        "date": "2030-01-01",
    }
    partner = client.post(
        "/api/v1/auth/public/service-area/check", json=service_payload
    )
    assert partner.status_code == 200
    assert partner.get_json()["reason_code"] == "PARTNER_REQUIRED"

    assert (
        client.post("/api/v1/auth/public/pre-request/draft", json={}).status_code == 400
    )
    monkeypatch.setattr(auth, "_public_cache_get", lambda _key: "{invalide")
    assert client.get("/api/v1/auth/public/pre-request/draft/test").status_code == 404
    assert (
        client.post(
            "/api/v1/auth/public/pre-request/consume", json={"draft_id": "test"}
        ).status_code
        == 200
    )

    monkeypatch.setattr("services.saferpay.config.saferpay_configured", lambda: False)
    assert (
        client.post(
            "/api/v1/auth/public/guest-booking/saferpay/initialize", json={}
        ).status_code
        == 503
    )
    assert (
        client.post(
            "/api/v1/auth/public/guest-booking/saferpay/assert", json={}
        ).status_code
        == 503
    )

    monkeypatch.setattr("services.saferpay.config.saferpay_configured", lambda: True)
    monkeypatch.setattr(
        auth, "_decode_guest_booking_status_token", lambda _token: ("guest-1", None)
    )
    monkeypatch.setattr(
        auth,
        "_public_cache_get",
        lambda _key: json.dumps({"guest_booking_id": "guest-1"}),
    )
    monkeypatch.setattr(
        "services.guest_saferpay.initialize_guest_saferpay",
        lambda **_kw: (_ for _ in ()).throw(ValueError("guest_booking_consumed")),
    )
    consumed = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/initialize",
        json={"status_token": "token"},
    )
    assert consumed.status_code == 409

    monkeypatch.setattr(
        "services.guest_saferpay.promote_guest_booking_after_saferpay",
        lambda **_kw: (_ for _ in ()).throw(ValueError("transaction invalide")),
    )
    invalid = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/assert",
        json={"status_token": "token"},
    )
    assert invalid.status_code == 400
    monkeypatch.setattr(
        "services.guest_saferpay.promote_guest_booking_after_saferpay",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("base indisponible")),
    )
    failed = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/assert",
        json={"status_token": "token"},
    )
    assert failed.status_code == 500


def test_gestionnaires_exceptions_routes_authentifiees(
    client, app, sample_user, monkeypatch
):
    with app.app_context():
        from flask_jwt_extended import create_access_token

        token = create_access_token(
            identity=str(sample_user.public_id),
            fresh=True,
            additional_claims={"aud": "atmr-api", "role": sample_user.role.value},
        )
    headers = {"Authorization": f"Bearer {token}"}
    monkeypatch.setenv("SECURITY_2FA_ENABLED", "true")
    monkeypatch.setattr(
        auth.user_repo,
        "find_by_public_id",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("dépôt")),
    )
    monkeypatch.setattr(auth.sentry_sdk, "capture_exception", lambda _error: None)
    monkeypatch.setattr(
        auth.APIErrorHandler,
        "handle_exception",
        lambda _error, _logger: ({"error": "interne"}, 500),
    )
    cases = [
        ("get", "/api/v1/auth/sessions", None),
        ("post", "/api/v1/auth/sessions/revoke-others", {}),
        ("post", "/api/v1/auth/totp/setup", {}),
        ("post", "/api/v1/auth/totp/verify", {"code": "123456"}),
        ("post", "/api/v1/auth/totp/disable", {"password": "secret"}),
        ("get", "/api/v1/auth/totp/status", None),
        ("post", "/api/v1/auth/totp/recovery-codes", {"code": "123456"}),
    ]
    for method, url, payload in cases:
        response = getattr(client, method)(url, json=payload, headers=headers)
        assert response.status_code == 500


def test_forgot_password_et_routes_exceptionnelles(
    client, app, sample_user, monkeypatch
):
    monkeypatch.setattr(auth.user_repo, "find_by_email", lambda _email: None)
    unknown = client.post(
        "/api/v1/auth/forgot-password", json={"email": "absent@example.test"}
    )
    assert unknown.status_code == 200

    monkeypatch.setattr(auth.user_repo, "find_by_email", lambda _email: sample_user)
    old_secret = app.config.get("SECRET_KEY")
    monkeypatch.setitem(app.config, "SECRET_KEY", None)
    no_secret = client.post(
        "/api/v1/auth/forgot-password", json={"email": sample_user.email}
    )
    assert no_secret.status_code == 500
    monkeypatch.setitem(app.config, "SECRET_KEY", old_secret)

    monkeypatch.setattr(
        auth.user_repo,
        "find_by_email",
        lambda _email: (_ for _ in ()).throw(RuntimeError("dépôt")),
    )
    monkeypatch.setattr(
        auth.APIErrorHandler,
        "handle_exception",
        lambda _error, _logger: ({"error": "interne"}, 500),
    )
    failed = client.post(
        "/api/v1/auth/forgot-password", json={"email": sample_user.email}
    )
    assert failed.status_code == 500

    monkeypatch.setattr(
        auth,
        "generate_csrf_token",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("csrf")),
    )
    csrf = client.get("/api/v1/auth/csrf-token")
    assert csrf.status_code == 500


def test_register_exceptions_terminales(client, monkeypatch):
    validated = {
        "username": "nouveau",
        "email": "nouveau@example.test",
        "password": "MotDePasse123!",
    }
    monkeypatch.setattr(auth, "validate_request", lambda *_a, **_kw: validated)
    monkeypatch.setattr(
        "security.password_policy.PasswordPolicyService.validate_password",
        lambda *_a, **_kw: None,
    )

    monkeypatch.setattr(
        auth,
        "RegisterUserUseCase",
        lambda: SimpleNamespace(
            execute=lambda _data: SimpleNamespace(
                success=True, user=None, error=None, status_code=201
            )
        ),
    )
    missing = client.post("/api/v1/auth/register", json=validated)
    assert missing.status_code == 500

    monkeypatch.setattr(
        auth,
        "RegisterUserUseCase",
        lambda: SimpleNamespace(
            execute=lambda _data: (_ for _ in ()).throw(
                auth.ValidationError({"user": ["invalide"]})
            )
        ),
    )
    validation = client.post("/api/v1/auth/register", json=validated)
    assert validation.status_code == 400

    monkeypatch.setattr(
        auth,
        "RegisterUserUseCase",
        lambda: SimpleNamespace(
            execute=lambda _data: (_ for _ in ()).throw(
                RuntimeError("erreur avec % de format")
            )
        ),
    )
    monkeypatch.setattr(auth.sentry_sdk, "capture_exception", lambda _error: None)
    failed = client.post("/api/v1/auth/register", json=validated)
    assert failed.status_code == 500


def _fresh_auth_headers(app, user):
    with app.app_context():
        from flask_jwt_extended import create_access_token

        token = create_access_token(
            identity=str(user.public_id),
            fresh=True,
            additional_claims={"aud": "atmr-api", "role": user.role.value},
        )
    return {"Authorization": f"Bearer {token}"}


def test_exceptions_fresh_me_sessions_et_mots_de_passe(
    client, app, sample_user, admin_headers, monkeypatch
):
    headers = _fresh_auth_headers(app, sample_user)
    monkeypatch.setattr(auth.sentry_sdk, "capture_exception", lambda _error: None)
    monkeypatch.setattr(
        auth.APIErrorHandler,
        "handle_exception",
        lambda _error, _logger: ({"error": "interne"}, 500),
    )

    monkeypatch.setattr(
        auth.user_repo,
        "find_by_public_id",
        lambda _pid: (_ for _ in ()).throw(RuntimeError("fresh")),
    )
    assert (
        client.post(
            "/api/v1/auth/fresh-token",
            json={"password": "secret"},
            headers=headers,
        ).status_code
        == 500
    )

    bootstrap_module = importlib.import_module(
        "application.auth_bootstrap.get_bootstrap_session_use_case"
    )
    monkeypatch.setattr(
        bootstrap_module,
        "GetBootstrapSessionUseCase",
        lambda: SimpleNamespace(
            execute=lambda: (_ for _ in ()).throw(RuntimeError("bootstrap"))
        ),
    )
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 500

    monkeypatch.setattr(
        auth.user_repo,
        "find_by_id",
        lambda _uid: (_ for _ in ()).throw(RuntimeError("admin")),
    )
    assert (
        client.post(
            f"/api/v1/auth/revoke-all-sessions/{sample_user.id}",
            json={},
            headers=admin_headers,
        ).status_code
        == 500
    )

    refresh_module = importlib.import_module("models.refresh_token")
    monkeypatch.setattr(
        refresh_module,
        "RefreshToken",
        SimpleNamespace(
            query=SimpleNamespace(
                get=lambda _sid: (_ for _ in ()).throw(RuntimeError("session"))
            )
        ),
    )
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: sample_user)
    assert client.delete("/api/v1/auth/sessions/42", headers=headers).status_code == 500

    class ThrowingQuery:
        def filter_by(self, **_kwargs):
            raise RuntimeError("utilisateur")

    monkeypatch.setattr(auth, "User", SimpleNamespace(query=ThrowingQuery()))
    assert (
        client.post(
            "/api/v1/auth/change-password",
            json={"new_password": "Nouveau123!"},
            headers=headers,
        ).status_code
        == 500
    )
    monkeypatch.setattr(
        auth,
        "_activation_serializer",
        lambda: (_ for _ in ()).throw(RuntimeError("serializer")),
    )
    assert (
        client.post(
            "/api/v1/auth/reset-password",
            json={"token": "token", "new_password": "Nouveau123!"},
        ).status_code
        == 500
    )


def test_gardes_totp_sessions_et_echec_challenge(client, app, sample_user, monkeypatch):
    headers = _fresh_auth_headers(app, sample_user)
    monkeypatch.setenv("SECURITY_2FA_ENABLED", "true")
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: None)
    cases = [
        ("post", "/api/v1/auth/totp/setup", {}),
        ("post", "/api/v1/auth/totp/verify", {"code": "123456"}),
        ("post", "/api/v1/auth/totp/disable", {"password": "secret"}),
        ("get", "/api/v1/auth/totp/status", None),
        ("post", "/api/v1/auth/totp/recovery-codes", {"code": "123456"}),
        ("get", "/api/v1/auth/sessions", None),
        ("post", "/api/v1/auth/sessions/revoke-others", {}),
    ]
    for method, url, payload in cases:
        response = getattr(client, method)(url, json=payload, headers=headers)
        assert response.status_code == 404

    sample_user.totp_enabled = True
    sample_user.totp_secret_encrypted = "secret"
    sample_user.recovery_codes_hash = "[]"
    sample_user.recovery_codes_remaining = 0
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: sample_user)
    assert (
        client.post("/api/v1/auth/totp/setup", json={}, headers=headers).status_code
        == 409
    )
    monkeypatch.setattr("security.totp_service.verify_totp_code", lambda *_args: False)
    assert (
        client.post(
            "/api/v1/auth/totp/verify",
            json={"code": "123456"},
            headers=headers,
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/v1/auth/totp/recovery-codes",
            json={"code": "123456"},
            headers=headers,
        ).status_code
        == 401
    )

    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda _token: (_ for _ in ()).throw(RuntimeError("jwt")),
    )
    monkeypatch.setattr(auth.sentry_sdk, "capture_exception", lambda _error: None)
    monkeypatch.setattr(
        auth.APIErrorHandler,
        "handle_exception",
        lambda _error, _logger: ({"error": "interne"}, 500),
    )
    challenge = client.post(
        "/api/v1/auth/totp/challenge",
        json={"temp_token": "temp", "code": "123456"},
    )
    assert challenge.status_code == 500


def test_invitation_exceptions_et_branches_metier(client, monkeypatch):
    monkeypatch.setattr(
        "application.institutions.invitation_service.hash_token",
        lambda _token: (_ for _ in ()).throw(RuntimeError("hash")),
    )
    assert client.get("/api/v1/auth/invite/token").status_code == 500
    assert (
        client.post(
            "/api/v1/auth/activate-account",
            json={"token": "token", "password": "MotDePasse123!"},
        ).status_code
        == 500
    )

    user = _activation_user(
        account_status="invited",
        invite_expires_at=datetime.now(UTC) + timedelta(hours=1),
        institution_id=12,
        institution_role="member",
        first_name="Jean",
        last_name="Test",
    )
    query = _Query(user)
    monkeypatch.setattr(auth, "User", SimpleNamespace(query=query))
    monkeypatch.setattr(
        "application.institutions.invitation_service.hash_token", lambda _token: "hash"
    )
    institution_module = importlib.import_module("models.institution")
    monkeypatch.setattr(
        institution_module,
        "Institution",
        SimpleNamespace(query=_Query(SimpleNamespace(name="Institution Test"))),
    )
    verified = client.get("/api/v1/auth/invite/token")
    assert verified.status_code == 200
    assert verified.get_json()["institution_name"] == "Institution Test"

    class PasswordPolicyError(Exception):
        pass

    password_policy = importlib.import_module("security.password_policy")
    monkeypatch.setattr(password_policy, "PasswordPolicyError", PasswordPolicyError)
    monkeypatch.setattr(
        password_policy.PasswordPolicyService,
        "validate_password",
        lambda *_a, **_kw: (_ for _ in ()).throw(PasswordPolicyError("faible")),
    )
    weak = client.post(
        "/api/v1/auth/activate-account",
        json={"token": "token", "password": "MotDePasse123!"},
    )
    assert weak.status_code == 400


def test_activation_gardes_utilisateur_et_politique(client, app, monkeypatch):
    session = _activation_session()
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)

    session.sms_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    expired = client.post(
        "/api/v1/auth/activation/verify-sms",
        json={"activation_session_id": session.activation_session_id, "code": "123456"},
    )
    assert expired.status_code == 400

    session.sms_expires_at = datetime.now(UTC) + timedelta(minutes=5)
    session.sms_attempts = auth.ACTIVATION_SMS_MAX_ATTEMPTS - 1
    locked = client.post(
        "/api/v1/auth/activation/verify-sms",
        json={"activation_session_id": session.activation_session_id, "code": "000000"},
    )
    assert locked.status_code == 429

    session.phone_verified_at = datetime.now(UTC)
    already_phone = client.post(
        "/api/v1/auth/activation/resend-sms",
        json={"activation_session_id": session.activation_session_id},
    )
    assert already_phone.status_code == 200

    session.phone_verified_at = None
    monkeypatch.setattr(auth, "User", SimpleNamespace(query=_Query(None)))
    missing_sms_user = client.post(
        "/api/v1/auth/activation/resend-sms",
        json={"activation_session_id": session.activation_session_id},
    )
    assert missing_sms_user.status_code == 404

    missing_update_user = client.post(
        "/api/v1/auth/activation/update-phone",
        json={
            "activation_session_id": session.activation_session_id,
            "phone": "+41790000000",
        },
    )
    assert missing_update_user.status_code == 404

    monkeypatch.setattr(auth, "User", SimpleNamespace(query=_Query(user)))
    monkeypatch.setitem(app.config, "TESTING", False)
    monkeypatch.setitem(app.config, "ENVIRONMENT", "development")
    session.last_sms_sent_at = None
    monkeypatch.setattr(
        auth, "_enforce_resend_policy", lambda **_kwargs: (True, None, 0)
    )
    monkeypatch.setattr(auth, "_send_activation_sms", lambda *_args: False)
    fallback_resend = client.post(
        "/api/v1/auth/activation/resend-sms",
        json={"activation_session_id": session.activation_session_id},
    )
    assert fallback_resend.status_code == 200

    session.phone_verified_at = datetime.now(UTC)
    confirmed = client.post(
        "/api/v1/auth/activation/update-phone",
        json={
            "activation_session_id": session.activation_session_id,
            "phone": "+41790000000",
        },
    )
    assert confirmed.status_code == 409

    session.phone_verified_at = None
    session.last_sms_sent_at = datetime.now(UTC)
    monkeypatch.setattr(
        auth, "_enforce_resend_policy", lambda **_kwargs: (False, "daily_limit", 0)
    )
    limited = client.post(
        "/api/v1/auth/activation/update-phone",
        json={
            "activation_session_id": session.activation_session_id,
            "phone": "+41790000000",
        },
    )
    assert limited.status_code == 429

    monkeypatch.setattr(
        auth, "_enforce_resend_policy", lambda **_kwargs: (True, None, 0)
    )
    fallback = client.post(
        "/api/v1/auth/activation/update-phone",
        json={
            "activation_session_id": session.activation_session_id,
            "phone": "+41790000000",
        },
    )
    assert fallback.status_code == 200


def test_statuts_publics_et_liens_invalides(client, app, sample_user, monkeypatch):
    assert client.get("/api/v1/auth/public/booking-status").status_code == 401
    monkeypatch.setattr(
        auth, "_load_booking_status_from_token", lambda _token: (None, None)
    )
    assert (
        client.get(
            "/api/v1/auth/public/booking-status", query_string={"token": "token"}
        ).status_code
        == 401
    )
    monkeypatch.setattr(
        auth, "_load_booking_status_from_token", lambda _token: (42, None)
    )
    monkeypatch.setattr(auth.db.session, "get", lambda *_args: None)
    assert (
        client.get(
            "/api/v1/auth/public/booking-status", query_string={"token": "token"}
        ).status_code
        == 404
    )

    monkeypatch.setattr(auth, "_public_cache_get", lambda _key: None)
    assert client.get("/api/v1/auth/public/pre-request/draft/absent").status_code == 404
    assert (
        client.post("/api/v1/auth/public/pre-request/consume", json={}).status_code
        == 400
    )

    headers = _fresh_auth_headers(app, sample_user)
    assert (
        client.post(
            "/api/v1/auth/public/guest-booking/link",
            json={},
            headers=headers,
        ).status_code
        == 400
    )
    serializer = SimpleNamespace(
        loads=lambda *_a, **_kw: (_ for _ in ()).throw(auth.SignatureExpired("expiré"))
    )
    monkeypatch.setattr(auth, "_public_link_serializer", lambda: serializer)
    assert (
        client.post(
            "/api/v1/auth/public/guest-booking/link",
            json={"status_token": "token"},
            headers=headers,
        ).status_code
        == 410
    )
    assert (
        client.get(
            "/api/v1/auth/public/guest-booking/status",
            query_string={"token": "token"},
        ).status_code
        == 410
    )


def test_reservation_invitee_gardes_restantes(client, app, sample_user, monkeypatch):
    required = {
        "departure": "Lausanne",
        "destination": "Pully",
        "date": "2030-01-01",
        "pickup_time": "10:00",
        "trip_type": "invalide",
    }
    monkeypatch.setattr(
        auth,
        "compute_public_guest_booking_price",
        lambda **_kwargs: {"ok": False, "error": "pricing_failed"},
    )
    failed_create = client.post(
        "/api/v1/auth/public/guest-booking/create", json=required
    )
    assert failed_create.status_code == 422

    assert (
        client.post(
            "/api/v1/auth/public/guest-booking/create",
            json={"departure": "Lausanne"},
        ).status_code
        == 400
    )
    failed_preview = client.post(
        "/api/v1/auth/public/guest-booking/preview", json=required
    )
    assert failed_preview.status_code == 422

    monkeypatch.setattr("services.saferpay.config.saferpay_configured", lambda: True)
    monkeypatch.setattr(
        auth, "_decode_guest_booking_status_token", lambda _token: ("guest-1", None)
    )
    mismatch = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/initialize",
        json={"status_token": "token", "guest_booking_id": "autre"},
    )
    assert mismatch.status_code == 400
    monkeypatch.setattr(auth, "_public_cache_get", lambda _key: None)
    missing = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/initialize",
        json={"status_token": "token"},
    )
    assert missing.status_code == 404

    monkeypatch.setattr(auth, "_public_cache_get", lambda _key: "{invalide")
    invalid_cache = client.post(
        "/api/v1/auth/public/guest-booking/saferpay/assert",
        json={"status_token": "token"},
    )
    assert invalid_cache.status_code == 404

    serializer = SimpleNamespace(
        loads=lambda *_a, **_kw: (_ for _ in ()).throw(auth.BadSignature("invalide"))
    )
    monkeypatch.setattr(auth, "_public_link_serializer", lambda: serializer)
    invalid_status = client.get(
        "/api/v1/auth/public/guest-booking/status",
        query_string={"token": "token"},
    )
    assert invalid_status.status_code == 401

    headers = _fresh_auth_headers(app, sample_user)
    invalid_link = client.post(
        "/api/v1/auth/public/guest-booking/link",
        json={"status_token": "token"},
        headers=headers,
    )
    assert invalid_link.status_code == 401

    serializer.loads = lambda *_a, **_kw: {}
    missing_identifier = client.post(
        "/api/v1/auth/public/guest-booking/link",
        json={"status_token": "token"},
        headers=headers,
    )
    assert missing_identifier.status_code == 401

    serializer.loads = lambda *_a, **_kw: {"guest_booking_id": "guest-1"}
    monkeypatch.setattr(auth, "_public_cache_get", lambda _key: None)
    missing_booking = client.post(
        "/api/v1/auth/public/guest-booking/link",
        json={"status_token": "token"},
        headers=headers,
    )
    assert missing_booking.status_code == 404

    monkeypatch.setattr(auth, "_public_cache_get", lambda _key: "{invalide")
    malformed_booking = client.post(
        "/api/v1/auth/public/guest-booking/link",
        json={"status_token": "token"},
        headers=headers,
    )
    assert malformed_booking.status_code == 404
