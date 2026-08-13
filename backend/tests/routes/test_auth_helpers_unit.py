"""Couverture unitaire des helpers critiques de ``routes.auth``."""

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from models import UserRole
from routes import auth


def test_detection_mobile_et_entetes_client(app):
    """Les marqueurs mobiles et métadonnées privilégient les nouveaux en-têtes."""
    with app.test_request_context(
        "/",
        headers={
            "X-Requested-With": "Expo",
            "X-Client-Platform": " ios ",
            "X-App-Version": " 2.1 ",
            "X-App-Build": " 42 ",
            "X-OS-Version": " 18.0 ",
        },
    ):
        assert auth._is_mobile_request() is True
        assert auth._resolve_client_platform() == "ios"
        assert auth._resolve_client_app_version() == "2.1"
        assert auth._resolve_client_app_build() == "42"
        assert auth._resolve_client_os_version() == "18.0"

    with app.test_request_context("/", headers={"User-Agent": "okhttp/4"}):
        assert auth._is_mobile_request() is True
    with app.test_request_context("/", headers={"X-Platform": "web"}):
        assert auth._is_mobile_request() is False
        assert auth._resolve_client_platform() == "web"


def test_metadonnees_et_nom_appareil(app):
    """Les noms d'application génériques sont écartés au profit du modèle."""
    headers = {
        "X-Device-Name": "LIRIE",
        "X-Device-Model": " Pixel 9 ",
        "X-Device-Manufacturer": " Google ",
        "X-Device-Type": " phone ",
        "X-Client-Platform": "android",
        "X-Active-Context-Id": "driver:4",
    }
    with app.test_request_context("/", headers=headers):
        meta = auth._resolve_device_session_metadata()
        assert meta.device_name is None
        assert meta.device_model == "Pixel 9"
        assert meta.device_manufacturer == "Google"
        assert meta.device_type == "phone"
        assert meta.context_id == "driver:4"
        assert auth._resolve_device_display_name() == "Pixel 9"

    with app.test_request_context("/"):
        assert auth._resolve_device_display_name() is None


@pytest.mark.parametrize(
    ("phone", "expected"),
    [
        (None, "inconnu"),
        ("", "inconnu"),
        ("7", "*"),
        ("12", "**"),
        ("+41 79 123 45 67", "+** *** *** 67"),
    ],
)
def test_masquage_telephone(phone, expected):
    assert auth._mask_phone(phone) == expected


def test_hash_statuts_et_identifiant_dossier(app, monkeypatch):
    assert len(auth._hash_plain_value("secret")) == 64
    assert auth._map_public_booking_status("ASSIGNED")[0] == "confirmed"
    assert auth._map_public_booking_status("started")[0] == "in_progress"
    assert auth._map_public_booking_status("done")[0] == "completed"
    assert auth._map_public_booking_status("rejected")[0] == "cancelled"
    assert auth._map_public_booking_status("new")[0] == "pending"
    assert auth._map_public_booking_status("autre")[0] == "unknown"
    assert auth._public_guest_booking_id_from_dossier_str("12") is None

    from models.enums import BookingCreatedVia

    with app.app_context():
        monkeypatch.setattr(
            auth.db.session,
            "get",
            lambda _model, booking_id: SimpleNamespace(
                created_via=BookingCreatedVia.PUBLIC_GUEST
            )
            if booking_id == 123
            else None,
        )
        assert auth._public_guest_booking_id_from_dossier_str("123") == 123
        assert auth._public_guest_booking_id_from_dossier_str("999") is None


def test_passwordless_selon_environnement(app, monkeypatch):
    with app.app_context():
        monkeypatch.setitem(app.config, "ENVIRONMENT", "production")
        assert auth._passwordless_allowed_in_environment() is False
        assert auth._passwordless_debug_code_enabled() is False
        monkeypatch.setitem(app.config, "ENVIRONMENT", "development")
        monkeypatch.setenv("PASSWORDLESS_DEBUG_CODE", "yes")
        assert auth._passwordless_allowed_in_environment() is True
        assert auth._passwordless_debug_code_enabled() is True


def test_activation_exigences_et_statut(app, monkeypatch):
    user = SimpleNamespace(email="a@test.ch", phone=None)
    monkeypatch.setattr(
        auth,
        "User",
        SimpleNamespace(query=SimpleNamespace(get=lambda _user_id: user)),
    )
    session = SimpleNamespace(
        user_id=1,
        email_verified_at=datetime.now(UTC),
        phone_verified_at=None,
        consumed_at=None,
        email_delivery_status="sent",
    )
    with app.app_context():
        assert auth._activation_channel_requirements(session) == (True, False)
        assert auth._activation_is_complete(session) is True
        status = auth._build_activation_status(session)
        assert status["is_complete"] is True
        assert status["is_finalized"] is False

        user.email = None
        assert auth._activation_channel_requirements(session) == (True, True)


def test_configuration_statuts_et_onboarding(app, monkeypatch):
    with app.app_context():
        monkeypatch.setitem(app.config, "FLAG_BOOL", "on")
        assert auth._bool_config("FLAG_BOOL") is True
        assert auth._bool_config("ABSENT", True) is True
        monkeypatch.setitem(
            app.config,
            "MOBILE_FEATURE_FLAGS",
            {"chat": True, "saferpay_enabled": False},
        )
        flags = auth._feature_flags_config()
        assert flags["chat"] is True
        assert flags["saferpay_enabled"] is False

    enum_status = SimpleNamespace(value="DISABLED")
    user = SimpleNamespace(account_status=enum_status, force_password_change=False)
    assert auth._account_status_value(user) == "disabled"
    assert auth._normalized_account_status(user) == "suspended"
    user.account_status = "invited"
    assert auth._normalized_account_status(user) == "inactive"
    assert auth._onboarding_required(user) is True
    assert auth._onboarding_reasons(user) == ["invited"]
    user.account_status = "pending_activation"
    user.force_password_change = True
    assert auth._onboarding_reasons(user) == [
        "force_password_change",
        "pending_activation",
    ]
    assert auth._must_complete_onboarding(user) is True


def test_permissions_et_resolution_contextes():
    assert "booking:create" in auth._permissions_for_context("client")
    assert "mission:read" in auth._permissions_for_context("driver")
    assert "company:dashboard:read" in auth._permissions_for_context("company")
    assert "institution:dashboard:read" in auth._permissions_for_context("institution")
    assert auth._permissions_for_context("admin") == []

    contexts = [
        {"context_id": "company:1", "context_type": "company", "is_default": True},
        {"context_id": "driver:2", "context_type": "driver"},
    ]
    assert auth._context_by_id(contexts, "driver:2") == contexts[1]
    assert auth._context_by_id(contexts, None) is None
    assert auth._is_company_driver_cross_context_switch(contexts[0], contexts[1])
    assert not auth._is_company_driver_cross_context_switch(None, contexts[1])
    assert auth._default_context_id(contexts) == "company:1"
    assert auth._default_context_id([{"context_id": "x"}]) == "x"
    assert auth._default_context_id([]) is None
    assert (
        auth._resolve_active_context_id(
            contexts=contexts, preferred_context_id="driver:2"
        )
        == "driver:2"
    )
    assert (
        auth._resolve_active_context_id(contexts=contexts, preferred_context_id="bad")
        == "company:1"
    )


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        (
            SimpleNamespace(
                account_status="pending_activation",
                role=UserRole.company,
                institution_id=None,
            ),
            False,
        ),
        (
            SimpleNamespace(
                account_status="active",
                role=UserRole.driver,
                driver=SimpleNamespace(is_active=False),
                institution_id=None,
            ),
            False,
        ),
        (
            SimpleNamespace(
                account_status="active",
                role=UserRole.client,
                clients=[SimpleNamespace(is_active=False)],
                institution_id=None,
            ),
            False,
        ),
        (
            SimpleNamespace(
                account_status="active",
                role=UserRole.client,
                clients=[SimpleNamespace(is_active=True)],
                institution_id=None,
            ),
            True,
        ),
        (
            SimpleNamespace(
                account_status="active",
                role=UserRole.company,
                institution_id=None,
            ),
            True,
        ),
    ],
)
def test_verification_profil_actif(monkeypatch, user, expected):
    monkeypatch.setattr(
        auth, "enforce_demo_user_access_validity", lambda _user: (True, None)
    )
    assert auth._check_user_profile_active(user)[0] is expected


def test_validation_refresh_token_branches(app, monkeypatch):
    repo = SimpleNamespace(
        find_by_public_id=lambda _public_id: None,
        find_model_by_public_id=lambda _public_id: None,
    )
    monkeypatch.setattr(auth, "user_repo", repo)
    monkeypatch.setattr(auth, "refresh_fail_closed_enabled", lambda: False)
    monkeypatch.setattr(auth, "update_token_last_used", lambda _token: None)

    with app.test_request_context("/"):
        monkeypatch.setattr(
            auth,
            "decode_token",
            lambda *_args, **_kwargs: {"sub": "u1", "type": "refresh"},
        )
        monkeypatch.setattr(auth, "is_token_revoked", lambda *_args, **_kwargs: False)
        assert auth._validate_refresh_token("ok") == ("u1", None)

        monkeypatch.setattr(
            auth,
            "decode_token",
            lambda *_args, **_kwargs: {"sub": "u1", "type": "access", "role": "client"},
        )
        assert auth._validate_refresh_token("access")[0] is None

        monkeypatch.setattr(
            auth, "decode_token", lambda *_args, **_kwargs: {"type": "refresh"}
        )
        assert auth._validate_refresh_token("sans-identite")[0] is None

        monkeypatch.setattr(
            auth,
            "decode_token",
            lambda *_args, **_kwargs: {"sub": "u1", "type": "refresh"},
        )
        monkeypatch.setattr(auth, "is_token_revoked", lambda *_args, **_kwargs: True)
        assert auth._validate_refresh_token("revoque")[0] is None

        monkeypatch.setattr(
            auth, "decode_token", MagicMock(side_effect=ValueError("jwt invalide"))
        )
        assert auth._validate_refresh_token("invalide")[0] is None


def test_expirations_et_remember_me(app, monkeypatch):
    with app.app_context():
        monkeypatch.setitem(
            app.config, "JWT_ACCESS_TOKEN_EXPIRES", timedelta(minutes=15)
        )
        monkeypatch.setitem(
            app.config, "JWT_MOBILE_ACCESS_TOKEN_EXPIRES", timedelta(hours=1)
        )
        monkeypatch.setitem(app.config, "JWT_REFRESH_TOKEN_EXPIRES", timedelta(days=10))
        assert auth._resolve_access_token_expires(False) == timedelta(minutes=15)
        assert auth._resolve_access_token_expires(True) == timedelta(hours=1)
        assert auth._resolve_refresh_token_expires(
            is_mobile_request=True, remember_me=False
        ) == timedelta(days=10)
        assert auth._resolve_refresh_token_expires(
            is_mobile_request=False, remember_me=True
        ) == timedelta(days=30)
        assert (
            auth._refresh_cookie_max_age(
                remember_me=True, refresh_expires_delta=timedelta(hours=2)
            )
            == 7200
        )
        assert (
            auth._refresh_cookie_max_age(
                remember_me=False, refresh_expires_delta=timedelta(hours=2)
            )
            is None
        )
        metadata = auth._access_expiry_metadata(timedelta(seconds=90))
        assert metadata["expires_in"] == 90
        assert str(metadata["access_expires_at"]).endswith("Z")

        assert (
            auth._resolve_remember_me_from_refresh_token("x", is_mobile_request=True)
            is False
        )
        monkeypatch.setattr(
            auth,
            "decode_token",
            lambda *_args, **_kwargs: {"remember_me": True, "iat": 0, "exp": 1},
        )
        assert (
            auth._resolve_remember_me_from_refresh_token("x", is_mobile_request=False)
            is True
        )
        monkeypatch.setattr(
            auth,
            "decode_token",
            lambda *_args, **_kwargs: {"iat": 0, "exp": 3600},
        )
        assert (
            auth._resolve_remember_me_from_refresh_token("x", is_mobile_request=False)
            is False
        )


def test_suppression_cookies_web(app, monkeypatch):
    response = MagicMock()
    appels = {"nombre": 0}

    def supprimer_cookie(*_args, **_kwargs):
        appels["nombre"] += 1
        if appels["nombre"] == 1:
            raise TypeError("ancienne signature")

    response.delete_cookie.side_effect = supprimer_cookie
    with app.app_context():
        configuration = {
            "COOKIE_ACCESS_TOKEN_NAME": "access",
            "COOKIE_REFRESH_TOKEN_NAME": "refresh",
            "COOKIE_DOMAIN": ".lirie.ch",
            "COOKIE_PATH": "/",
            "COOKIE_SECURE": True,
            "COOKIE_HTTP_ONLY": True,
            "COOKIE_SAME_SITE": "Lax",
        }
        for nom, valeur in configuration.items():
            monkeypatch.setitem(app.config, nom, valeur)
        auth._clear_web_auth_cookies(response)
    assert response.set_cookie.call_count == 6
    assert response.delete_cookie.call_count == 7


def test_jour_utc_et_politique_renvoi(monkeypatch):
    now = datetime.now(UTC)
    assert auth._is_same_utc_day(now, now)
    monkeypatch.setattr(auth, "ACTIVATION_RESEND_COOLDOWN_SECONDS", 60)
    monkeypatch.setattr(auth, "ACTIVATION_RESEND_DAILY_LIMIT", 2)
    allowed, reason, retry = auth._enforce_resend_policy(
        last_sent_at=now - timedelta(seconds=10), resend_count=0
    )
    assert (allowed, reason) == (False, "cooldown")
    assert retry > 0
    assert auth._enforce_resend_policy(
        last_sent_at=now.replace(hour=0, minute=0, second=0, microsecond=0),
        resend_count=2,
    ) == (False, "daily_limit", 0)
    assert auth._enforce_resend_policy(
        last_sent_at=now - timedelta(days=1), resend_count=9
    ) == (True, None, 0)


def test_helpers_branches_rares_et_reinitialisation(app, monkeypatch):
    with app.test_request_context(
        "/",
        headers={"X-Client-Platform": "ios", "X-Device-Manufacturer": "Fairphone"},
    ):
        assert auth._is_mobile_request() is True
        assert auth._resolve_device_display_name() == "Fairphone"

    with app.app_context():
        monkeypatch.setitem(app.config, "SECRET_KEY", None)
        with pytest.raises(RuntimeError):
            auth._activation_serializer()
        with pytest.raises(RuntimeError):
            auth._public_link_serializer()

        monkeypatch.setenv("BOOKING_STATUS_TOKEN_TTL_SECONDS", "invalide")
        monkeypatch.setenv("PUBLIC_GUEST_BOOKING_TTL_SECONDS", "invalide")
        monkeypatch.setenv("PASSWORDLESS_OTP_TTL_SECONDS", "invalide")
        assert auth._resolve_booking_status_token_ttl_seconds() == 3600
        assert auth._resolve_guest_booking_ttl_seconds() == 604800
        assert auth._resolve_passwordless_otp_ttl_seconds() == 600

    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("invalide")),
    )
    with app.test_request_context("/"):
        assert (
            auth._resolve_remember_me_from_refresh_token(
                "token", is_mobile_request=False
            )
            is False
        )
    monkeypatch.setattr(
        auth,
        "decode_token",
        lambda *_args, **_kwargs: {
            "iat": 0,
            "exp": 30 * 24 * 3600,
        },
    )
    with app.test_request_context("/"):
        assert (
            auth._resolve_remember_me_from_refresh_token(
                "token", is_mobile_request=False
            )
            is True
        )

    redis_values = iter([b"bytes", "texte"])
    redis = SimpleNamespace(
        get=lambda _key: next(redis_values),
        setex=lambda *_args: None,
        delete=lambda _key: None,
    )
    monkeypatch.setattr(auth, "redis_client", redis)
    assert auth._public_cache_get("a") == "bytes"
    assert auth._public_cache_get("b") == "texte"
    auth._public_cache_setex("c", 60, "valeur")
    auth._public_cache_delete("c")
    monkeypatch.setattr(auth, "redis_client", None)

    serializer = SimpleNamespace(
        loads=MagicMock(side_effect=auth.SignatureExpired("expiré"))
    )
    monkeypatch.setattr(auth, "_public_link_serializer", lambda: serializer)
    assert auth._decode_guest_booking_status_token("token") == (None, "token_expired")
    serializer.loads.side_effect = auth.BadSignature("invalide")
    monkeypatch.setattr(
        auth, "_public_guest_booking_id_from_dossier_str", lambda _t: None
    )
    assert auth._load_booking_status_from_token("token") == (None, "invalid")
    serializer.loads.side_effect = None
    serializer.loads.return_value = []
    assert auth._load_booking_status_from_token("token") == (None, "invalid")
    serializer.loads.return_value = {}
    assert auth._load_booking_status_from_token("token") == (None, "invalid")
    serializer.loads.return_value = {"booking_id": "invalide"}
    assert auth._load_booking_status_from_token("token") == (None, "invalid")

    user = SimpleNamespace(
        id=7,
        force_password_change=True,
        token_version=1,
        password_expires_at=datetime.now(UTC),
        temporary_password_created_at=datetime.now(UTC),
        first_login_completed_at=None,
        institution_id=3,
        authentication_method="username",
        set_password=lambda password: setattr(user, "password", password),
    )
    monkeypatch.setattr(
        "security.password_policy.PasswordPolicyService.validate_password",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "models.institution_user_audit_event.InstitutionUserAuditEvent.record",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "security.mobile_device_session_service.revoke_user_security_sessions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("révocation")),
    )
    monkeypatch.setattr(auth.db.session, "commit", lambda: None)
    with app.test_request_context("/"):
        body, status = auth._reset_user_password_with_policy(
            user, f"AtmrTest-{uuid.uuid4().hex[:10]}-Aa1!"
        )
    assert status == 200
    assert body["require_relogin"] is True
    assert user.first_login_completed_at is not None
