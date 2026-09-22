"""Contrat du provider SMS (Programmable Messaging, sans appel réseau)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from services.notifications.phone_e164 import normalize_e164_phone
from services.notifications.sms import (
    ERROR_AUTH,
    ERROR_CONFIG,
    ERROR_DESTINATION,
    ERROR_DISABLED,
    ERROR_SENDER,
    ERROR_SUCCESS,
    describe_sms_config,
    send_sms_notification,
)


def _enable_ready_config(monkeypatch):
    monkeypatch.setenv("SMS_NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx89")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token-test")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15005550006")
    monkeypatch.delenv("TWILIO_MESSAGING_SERVICE_SID", raising=False)


def _install_fake_twilio(monkeypatch, client_cls):
    fake_rest = types.ModuleType("twilio.rest")
    fake_rest.Client = client_cls
    monkeypatch.setitem(sys.modules, "twilio", types.ModuleType("twilio"))
    monkeypatch.setitem(sys.modules, "twilio.rest", fake_rest)


def test_normalize_swiss_e164_philippe_number():
    assert normalize_e164_phone("+41768190077") == "+41768190077"
    assert normalize_e164_phone("0768190077") == "+41768190077"
    assert normalize_e164_phone("0041768190077") == "+41768190077"
    assert normalize_e164_phone("41768190077") == "+41768190077"
    assert normalize_e164_phone("abc") is None


def test_sms_disabled_does_not_call_twilio(monkeypatch):
    monkeypatch.setenv("SMS_NOTIFICATIONS_ENABLED", "false")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx89")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token-test")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15005550006")
    called = {"create": 0}

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        @property
        def messages(self):
            return self

        def create(self, **_kwargs):
            called["create"] += 1
            return SimpleNamespace(sid="SM1234", status="queued")

    _install_fake_twilio(monkeypatch, _Client)
    result = send_sms_notification(
        "+41768190077", "secret-otp-999999", "activation_signup"
    )
    assert result["ok"] is False
    assert result["disabled"] is True
    assert result["error_class"] == ERROR_DISABLED
    assert called["create"] == 0
    assert "999999" not in str(result)


def test_sms_enabled_calls_provider_once(monkeypatch):
    _enable_ready_config(monkeypatch)
    called = {"create": 0, "to": None, "from_": None}

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        @property
        def messages(self):
            return self

        def create(self, **kwargs):
            called["create"] += 1
            called["to"] = kwargs.get("to")
            called["from_"] = kwargs.get("from_")
            return SimpleNamespace(
                sid="SMabcdef1234567890abcdef1234567890", status="queued"
            )

    _install_fake_twilio(monkeypatch, _Client)
    result = send_sms_notification(
        "+41768190077", "code-ne-pas-logger", "activation_signup"
    )
    assert result["ok"] is True
    assert result["error_class"] == ERROR_SUCCESS
    assert called["create"] == 1
    assert called["to"] == "+41768190077"
    assert called["from_"] == "+15005550006"
    assert result["message_sid"]
    assert "SMabcdef1234567890abcdef1234567890" not in result["message_sid"]
    assert result["destination_masked"] != "+41768190077"


def test_sms_missing_sender_is_sender_error(monkeypatch):
    monkeypatch.setenv("SMS_NOTIFICATIONS_ENABLED", "true")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx89")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "token-test")
    monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)
    monkeypatch.delenv("TWILIO_MESSAGING_SERVICE_SID", raising=False)
    result = send_sms_notification("+41768190077", "otp", "activation_signup")
    assert result["ok"] is False
    assert result["error_class"] == ERROR_SENDER


def test_sms_missing_credentials_is_config_error(monkeypatch):
    monkeypatch.setenv("SMS_NOTIFICATIONS_ENABLED", "true")
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15005550006")
    result = send_sms_notification("+41768190077", "otp", "activation_signup")
    assert result["ok"] is False
    assert result["error_class"] == ERROR_CONFIG


def test_sms_invalid_destination(monkeypatch):
    _enable_ready_config(monkeypatch)
    result = send_sms_notification("12", "otp", "activation_signup")
    assert result["ok"] is False
    assert result["error_class"] == ERROR_DESTINATION


def test_sms_twilio_auth_error_classified(monkeypatch):
    _enable_ready_config(monkeypatch)

    class _AuthError(Exception):
        code = 20003
        status = 401

    class _Client:
        def __init__(self, *_args, **_kwargs):
            pass

        @property
        def messages(self):
            return self

        def create(self, **_kwargs):
            raise _AuthError("Authenticate")

    _install_fake_twilio(monkeypatch, _Client)
    result = send_sms_notification("+41768190077", "otp", "activation_signup")
    assert result["ok"] is False
    assert result["error_class"] == ERROR_AUTH
    assert result["provider_error_code"] == "20003"


def test_describe_sms_config_never_exposes_secrets(monkeypatch):
    _enable_ready_config(monkeypatch)
    snapshot = describe_sms_config()
    rendered = str(snapshot)
    assert "token-test" not in rendered
    assert "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx89" not in rendered
    assert snapshot["twilio_account_sid"] == "PRESENT"
    assert snapshot["twilio_auth_token"] == "PRESENT"
    assert snapshot["twilio_mode"] == "PROGRAMMABLE_MESSAGING"
