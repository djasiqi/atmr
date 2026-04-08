import os

import pytest

from services.worldline import return_url as ru


def test_allowed_prefixes_from_env(monkeypatch):
    monkeypatch.setenv("CLIENT_WEB_BASE_URL", "https://app.example.com")
    monkeypatch.setenv("WORLDLINE_ALLOWED_RETURN_URL_PREFIXES", "https://legacy.example.org")
    p = ru.allowed_return_url_prefixes()
    assert "https://app.example.com" in p
    assert "https://legacy.example.org" in p


def test_validate_override_ok(monkeypatch):
    monkeypatch.setenv("CLIENT_WEB_BASE_URL", "https://app.example.com")
    u = ru.validate_return_url_override(
        "https://app.example.com/client/payment/worldline/return?bookingId=1"
    )
    assert "bookingId=1" in u


def test_validate_override_rejected(monkeypatch):
    monkeypatch.setenv("CLIENT_WEB_BASE_URL", "https://app.example.com")
    with pytest.raises(ValueError, match="non autorisée"):
        ru.validate_return_url_override("https://evil.com/phish")


def test_validate_override_requires_config(monkeypatch):
    monkeypatch.delenv("CLIENT_WEB_BASE_URL", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("WORLDLINE_ALLOWED_RETURN_URL_PREFIXES", raising=False)
    with pytest.raises(ValueError, match="CLIENT_WEB_BASE_URL"):
        ru.validate_return_url_override("https://any.com/x")


def test_https_required(monkeypatch):
    monkeypatch.setenv("CLIENT_WEB_BASE_URL", "https://app.example.com")
    monkeypatch.setenv("WORLDLINE_RETURN_URL_REQUIRE_HTTPS", "true")
    with pytest.raises(ValueError, match="https"):
        ru.validate_return_url_override("http://app.example.com/return")


def test_default_return_url(monkeypatch):
    monkeypatch.setenv("CLIENT_WEB_BASE_URL", "https://app.example.com/")
    d = ru.default_worldline_return_url(42)
    assert d.startswith("https://app.example.com")
    assert "bookingId=42" in d
