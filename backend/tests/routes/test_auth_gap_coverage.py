"""Complément de couverture ``routes/auth.py`` (seuil 95 %)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from werkzeug.exceptions import BadRequest

from models.enums import BookingCreatedVia, UserRole
from routes import auth
from security.password_policy import PasswordPolicyError


def test_display_name_et_otp_et_token_version(app):
    with app.test_request_context("/", headers={"X-Device-Name": "iPhone de Ada"}):
        assert auth._resolve_device_display_name() == "iPhone de Ada"
    code = auth._create_passwordless_otp_code()
    assert len(code) == 6
    assert code.isdigit()
    assert auth._user_token_version(SimpleNamespace()) == 0
    assert auth._user_token_version(SimpleNamespace(token_version=4)) == 4


def test_guest_token_vide_et_dossier_booking(app, monkeypatch):
    serializer = SimpleNamespace(loads=lambda *_a, **_k: {"guest_booking_id": "  "})
    monkeypatch.setattr(auth, "_public_link_serializer", lambda: serializer)
    assert auth._decode_guest_booking_status_token("tok") == (None, "token_invalid")

    monkeypatch.setattr(auth.db.session, "get", lambda *_a, **_k: None)
    assert auth._public_guest_booking_id_from_dossier_str("12345") is None

    guest = SimpleNamespace(
        created_via=SimpleNamespace(value=BookingCreatedVia.PUBLIC_GUEST.value)
    )
    monkeypatch.setattr(auth.db.session, "get", lambda *_a, **_k: guest)
    assert auth._public_guest_booking_id_from_dossier_str("12345") == 12345

    other = SimpleNamespace(created_via="dispatcher")
    monkeypatch.setattr(auth.db.session, "get", lambda *_a, **_k: other)
    assert auth._public_guest_booking_id_from_dossier_str("12345") is None

    expired = SimpleNamespace(
        loads=MagicMock(side_effect=auth.SignatureExpired("expiré"))
    )
    monkeypatch.setattr(auth, "_public_link_serializer", lambda: expired)
    monkeypatch.setattr(auth, "_public_cache_get", lambda _k: None)
    assert auth._load_booking_status_from_token("tok") == (None, "expired")


def test_reset_password_politique_et_audit(app, monkeypatch):
    user = SimpleNamespace(
        id=1,
        force_password_change=False,
        token_version=0,
        set_password=lambda _p: None,
    )
    monkeypatch.setattr(
        "security.password_policy.PasswordPolicyService.validate_password",
        lambda *_a, **_k: (_ for _ in ()).throw(PasswordPolicyError("trop court")),
    )
    with app.test_request_context("/"):
        resp = auth._reset_user_password_with_policy(user, "x")
    assert resp[1] == 400

    inst_user = SimpleNamespace(
        id=2,
        force_password_change=True,
        token_version=1,
        password_expires_at=None,
        temporary_password_created_at=None,
        first_login_completed_at=None,
        institution_id=9,
        authentication_method="username",
        set_password=lambda _p: None,
    )
    monkeypatch.setattr(
        "security.password_policy.PasswordPolicyService.validate_password",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "models.institution_user_audit_event.InstitutionUserAuditEvent.record",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("audit")),
    )
    monkeypatch.setattr(
        "security.mobile_device_session_service.revoke_user_security_sessions",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "security.security_metrics.security_token_invalidations_total.labels",
        lambda **_k: SimpleNamespace(inc=lambda: None),
        raising=False,
    )
    monkeypatch.setattr(
        "security.token_blacklist.revoke_token",
        lambda: (_ for _ in ()).throw(RuntimeError("revoke")),
    )
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    with app.test_request_context("/"):
        _body, status = auth._reset_user_password_with_policy(
            inst_user, "AtmrTest-Aa1!"
        )
    assert status == 200
    assert inst_user.first_login_completed_at is not None


def test_send_activation_sms_ok_et_ko(monkeypatch):
    monkeypatch.setattr(
        "services.notifications.sms.send_sms_notification",
        lambda **_k: {"ok": True},
    )
    assert (
        auth._send_activation_sms(SimpleNamespace(phone="+41791234567"), "123456")
        is True
    )
    monkeypatch.setattr(
        "services.notifications.sms.send_sms_notification",
        lambda **_k: {"ok": False},
    )
    assert (
        auth._send_activation_sms(SimpleNamespace(phone="+41791234567"), "123456")
        is False
    )


def test_user_schema_et_bool_config(app, monkeypatch):
    loaded = auth.UserSchema().load(
        {
            "username": "ada",
            "email": "ada@test.ch",
            "password": "password123",
        }
    )
    assert loaded["username"] == "ada"
    with app.app_context():
        monkeypatch.delitem(app.config, "GAP_FLAG", raising=False)
        monkeypatch.setenv("GAP_FLAG", "yes")
        assert auth._bool_config("GAP_FLAG") is True
        monkeypatch.delenv("GAP_FLAG_NONE", raising=False)
        assert auth._bool_config("GAP_FLAG_NONE", False) is False


def test_login_json_badrequest_et_exception(app, monkeypatch):
    with app.test_request_context("/api/v1/auth/login", method="POST"):
        monkeypatch.setattr(
            "flask.Request.get_json",
            lambda *_a, **_k: (_ for _ in ()).throw(BadRequest("json")),
        )
        resp = auth._login_post_body()
    assert resp[1] == 400

    with app.test_request_context("/api/v1/auth/login", method="POST"):
        monkeypatch.setattr(
            "flask.Request.get_json",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("autre")),
        )
        with pytest.raises(RuntimeError, match="autre"):
            auth._login_post_body()


def test_login_echec_audit_et_compte_en_attente(app, monkeypatch):
    class _UC:
        def execute(self, _inp):
            return SimpleNamespace(user=None, error={"error": "invalid_credentials"})

    monkeypatch.setattr(auth, "AuthenticateUserUseCase", lambda: _UC())
    monkeypatch.setattr(
        auth.AuditLogger,
        "log_action",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("audit")),
    )
    monkeypatch.setattr(
        "security.security_alerts.SecurityAlertService.record_login_failure",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("alert")),
    )
    with app.test_request_context(
        "/api/v1/auth/login",
        method="POST",
        json={"email": "a@test.ch", "password": "password123"},
    ):
        resp = auth._login_post_body()
    assert resp[1] == 401

    pending = SimpleNamespace(
        id=9,
        email="p@test.ch",
        phone="+41790000000",
        account_status="pending_activation",
        role=UserRole.client,
        clients=[],
        driver=None,
        institution_id=None,
    )
    session = SimpleNamespace(activation_session_id="act-99")
    q = MagicMock()
    q.filter_by.return_value.order_by.return_value.first.return_value = session

    class _PendingUC:
        def execute(self, _inp):
            return SimpleNamespace(user=pending, error=None)

    monkeypatch.setattr(auth, "AuthenticateUserUseCase", lambda: _PendingUC())
    monkeypatch.setattr(
        auth,
        "ActivationSession",
        SimpleNamespace(
            query=q,
            created_at=SimpleNamespace(desc=lambda: "created_at"),
        ),
    )
    with app.test_request_context(
        "/api/v1/auth/login",
        method="POST",
        json={"email": "p@test.ch", "password": "password123"},
    ):
        body, code = auth._login_post_body()
    assert code == 403
    assert body["reason"] == "account_pending_activation"
    assert body["activation_session_id"] == "act-99"

    disabled = SimpleNamespace(
        id=10,
        email="d@test.ch",
        phone=None,
        account_status="disabled",
        role=UserRole.client,
        clients=[SimpleNamespace(is_active=False)],
        driver=None,
        institution_id=None,
    )

    class _DisabledUC:
        def execute(self, _inp):
            return SimpleNamespace(user=disabled, error=None)

    monkeypatch.setattr(auth, "AuthenticateUserUseCase", lambda: _DisabledUC())
    with app.test_request_context(
        "/api/v1/auth/login",
        method="POST",
        json={"email": "d@test.ch", "password": "password123"},
    ):
        body2, code2 = auth._login_post_body()
    assert code2 == 403
    assert body2["reason"] == "account_disabled"


def test_bootstrap_context_demo_et_refresh_limits(app, monkeypatch):
    assert auth._context_by_id([{"context_id": "a"}], "missing") is None
    assert auth._context_by_id([{"context_id": "a"}], None) is None

    class _Ensure:
        def execute(self, _user):
            return SimpleNamespace(created=True)

    monkeypatch.setattr(
        "application.companies.drivers.ensure_company_operator_driver.EnsureCompanyOperatorDriverUseCase",
        _Ensure,
    )
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    reloaded = SimpleNamespace(id=2)
    monkeypatch.setattr(auth, "_load_user_for_bootstrap", lambda _pid: reloaded)
    assert (
        auth._prepare_user_for_bootstrap(SimpleNamespace(public_id="abc")) is reloaded
    )
    monkeypatch.setattr(auth, "_load_user_for_bootstrap", lambda _pid: None)
    original = SimpleNamespace(public_id="abc")
    assert auth._prepare_user_for_bootstrap(original) is original

    monkeypatch.setattr(
        auth, "enforce_demo_user_access_validity", lambda _u: (False, None)
    )
    demo = SimpleNamespace(
        account_status="active",
        role=UserRole.company,
        driver=None,
        clients=None,
        institution_id=None,
        force_password_change=False,
    )
    ok, msg = auth._check_user_profile_active(demo)
    assert ok is False
    assert "démo" in (msg or "").lower() or msg is not None

    assert (
        auth._resolve_max_active_refresh_tokens(SimpleNamespace(role=UserRole.client))
        >= 1
    )


def test_validate_refresh_store_et_last_used(app, monkeypatch):
    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda _t, allow_expired=False: {"sub": "pub-1", "type": "refresh"},
    )
    monkeypatch.setattr(auth.user_repo, "find_by_public_id", lambda _pid: None)
    monkeypatch.setattr(
        auth,
        "is_token_revoked",
        lambda *_a, **_k: (_ for _ in ()).throw(
            auth.RefreshStoreUnavailableError("down")
        ),
    )
    with app.test_request_context("/"):
        uid, err = auth._validate_refresh_token("rt")
    assert uid is None
    assert err is not None
    assert err.get("_http_status") == 503

    monkeypatch.setattr(
        auth,
        "is_token_revoked",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("check")),
    )
    monkeypatch.setattr(auth, "refresh_fail_closed_enabled", lambda: True)
    with app.test_request_context("/"):
        uid2, err2 = auth._validate_refresh_token("rt")
    assert uid2 is None
    assert err2 is not None
    assert err2.get("_http_status") == 503

    monkeypatch.setattr(auth, "refresh_fail_closed_enabled", lambda: False)
    monkeypatch.setattr(auth, "is_token_revoked", lambda *_a, **_k: False)
    monkeypatch.setattr(
        auth,
        "update_token_last_used",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("last_used")),
    )
    with app.test_request_context("/"):
        uid3, err3 = auth._validate_refresh_token("rt")
    assert uid3 == "pub-1"
    assert err3 is None
