"""Contrat d'activation SMS : erreurs, OTP, cooldown, idempotence."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from routes import auth
from tests.routes.test_auth_activation_coverage import (
    _activation_session,
    _activation_user,
    _patch_activation_models,
)


def test_resend_sms_disabled_returns_503_not_502(client, monkeypatch):
    session = _activation_session()
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    monkeypatch.setattr(
        auth,
        "_send_activation_sms",
        lambda *_a, **_k: {
            "ok": False,
            "error_class": "DISABLED",
            "disabled": True,
        },
    )
    response = client.post(
        "/api/v1/auth/activation/resend-sms",
        json={"activation_session_id": session.activation_session_id},
    )
    assert response.status_code == 503
    body = response.get_json()
    assert body["error"] == "sms_unavailable"
    assert "temporairement" in body["message"].lower()


def test_resend_sms_enabled_calls_provider_once(client, monkeypatch):
    session = _activation_session()
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    calls = {"n": 0}

    def _send(_user, _code):
        calls["n"] += 1
        return {"ok": True, "error_class": "SUCCESS"}

    monkeypatch.setattr(auth, "_send_activation_sms", _send)
    response = client.post(
        "/api/v1/auth/activation/resend-sms",
        json={"activation_session_id": session.activation_session_id},
    )
    assert response.status_code == 200
    assert calls["n"] == 1
    assert session.last_sms_sent_at is not None


def test_resend_sms_immediate_is_rate_limited(client, monkeypatch):
    session = _activation_session(last_sms_sent_at=datetime.now(UTC))
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    called = {"n": 0}
    monkeypatch.setattr(
        auth,
        "_send_activation_sms",
        lambda *_a, **_k: called.__setitem__("n", called["n"] + 1)
        or {"ok": True, "error_class": "SUCCESS"},
    )
    response = client.post(
        "/api/v1/auth/activation/resend-sms",
        json={"activation_session_id": session.activation_session_id},
    )
    assert response.status_code == 429
    assert response.get_json()["error"] == "rate_limited"
    assert called["n"] == 0


def test_verify_sms_wrong_code_does_not_activate(client, monkeypatch):
    session = _activation_session()
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    response = client.post(
        "/api/v1/auth/activation/verify-sms",
        json={"activation_session_id": session.activation_session_id, "code": "000000"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_credentials"
    assert session.phone_verified_at is None
    assert user.account_status == "pending_activation"


def test_verify_sms_expired_code_refused(client, monkeypatch):
    session = _activation_session(
        sms_expires_at=datetime.now(UTC) - timedelta(seconds=1)
    )
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    response = client.post(
        "/api/v1/auth/activation/verify-sms",
        json={"activation_session_id": session.activation_session_id, "code": "123456"},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "token_expired"
    assert session.phone_verified_at is None


def test_verify_sms_correct_code_then_finalize(client, monkeypatch):
    session = _activation_session(email_verified_at=datetime.now(UTC))
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    verify = client.post(
        "/api/v1/auth/activation/verify-sms",
        json={"activation_session_id": session.activation_session_id, "code": "123456"},
    )
    assert verify.status_code == 200
    assert session.phone_verified_at is not None

    finalize = client.post(
        "/api/v1/auth/activation/finalize",
        json={"activation_session_id": session.activation_session_id},
    )
    assert finalize.status_code == 200
    assert user.account_status == "active"
    assert session.consumed_at is not None
    assert user.clients[0].is_active is True


def test_verify_sms_double_validation_is_idempotent(client, monkeypatch):
    now = datetime.now(UTC)
    session = _activation_session(phone_verified_at=now)
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    first = client.post(
        "/api/v1/auth/activation/verify-sms",
        json={"activation_session_id": session.activation_session_id, "code": "123456"},
    )
    second = client.post(
        "/api/v1/auth/activation/verify-sms",
        json={"activation_session_id": session.activation_session_id, "code": "123456"},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert session.phone_verified_at == now


def test_update_phone_invalidates_old_otp(client, monkeypatch):
    session = _activation_session()
    old_hash = session.sms_code_hash
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    monkeypatch.setattr(auth, "_generate_sms_otp", lambda: "654321")
    monkeypatch.setattr(
        auth,
        "_send_activation_sms",
        lambda *_a, **_k: {"ok": True, "error_class": "SUCCESS"},
    )
    response = client.post(
        "/api/v1/auth/activation/update-phone",
        json={
            "activation_session_id": session.activation_session_id,
            "phone": "+41768190077",
        },
    )
    assert response.status_code == 200
    assert user.phone == "+41768190077"
    assert session.sms_code_hash != old_hash
    assert session.sms_code_hash == auth._hash_plain_value("654321")
    assert session.phone_verified_at is None


def test_update_phone_rejects_invalid_number(client, monkeypatch):
    session = _activation_session()
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(
        auth,
        "validate_request",
        lambda _schema, data, **_kw: {
            "activation_session_id": session.activation_session_id,
            "phone": "12345ab",
        },
    )
    response = client.post(
        "/api/v1/auth/activation/update-phone",
        json={
            "activation_session_id": session.activation_session_id,
            "phone": "12345ab",
        },
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_phone"
    assert user.phone == "+41791234567"


def test_provider_failure_does_not_verify_phone(client, monkeypatch):
    session = _activation_session()
    user = _activation_user()
    _patch_activation_models(monkeypatch, session, user)
    monkeypatch.setattr(auth, "validate_request", lambda _schema, data, **_kw: data)
    monkeypatch.setattr(
        auth,
        "_send_activation_sms",
        lambda *_a, **_k: {
            "ok": False,
            "error_class": "DELIVERY_FAILURE",
            "provider_error_code": "30008",
        },
    )
    response = client.post(
        "/api/v1/auth/activation/resend-sms",
        json={"activation_session_id": session.activation_session_id},
    )
    assert response.status_code == 503
    assert response.get_json()["error"] == "sms_provider_unavailable"
    assert session.phone_verified_at is None
    assert user.account_status == "pending_activation"


def test_send_activation_sms_missing_phone_is_destination_error():
    result = auth._send_activation_sms(SimpleNamespace(phone=None), "123456")
    assert auth._sms_send_succeeded(result) is False
    assert result["error_class"] == "DESTINATION_ERROR"
