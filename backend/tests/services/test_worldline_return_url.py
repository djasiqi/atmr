"""Return URL Worldline : même origine structurée, pas de préfixe chaîne."""

from __future__ import annotations

from urllib.parse import urlsplit

import pytest

from services.worldline import return_url as wl


def _isolate_checkout_base(monkeypatch) -> None:
    monkeypatch.delenv("CLIENT_WEB_BASE_URL", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("WORLDLINE_RETURN_URL_REQUIRE_HTTPS", raising=False)
    monkeypatch.setenv(
        "WORLDLINE_ALLOWED_RETURN_URL_PREFIXES", "https://checkout.example.org"
    )


def test_validate_return_url_override_ok(monkeypatch):
    _isolate_checkout_base(monkeypatch)
    accepted = wl.validate_return_url_override(
        "https://checkout.example.org/client/payment/worldline/return?x=1"
    )
    parts = urlsplit(accepted)
    assert parts.scheme == "https"
    assert parts.hostname == "checkout.example.org"
    assert parts.path == "/client/payment/worldline/return"
    assert parts.username is None


def test_validate_return_url_rejects_unknown_host(monkeypatch):
    _isolate_checkout_base(monkeypatch)
    with pytest.raises(ValueError, match="return_url non autorisée"):
        wl.validate_return_url_override("https://evil.com/phish")


@pytest.mark.parametrize(
    "candidate",
    [
        "https://checkout.example.org",
        "https://checkout.example.org/cb?x=1",
        "https://CHECKOUT.EXAMPLE.ORG/cb",
        "https://checkout.example.org:443/cb",
        "https://checkout.example.org./cb",
    ],
)
def test_validate_return_url_accepts_same_origin(monkeypatch, candidate):
    _isolate_checkout_base(monkeypatch)
    accepted = wl.validate_return_url_override(candidate)
    parts = urlsplit(accepted)
    assert parts.scheme == "https"
    assert parts.hostname.rstrip(".") == "checkout.example.org"
    assert parts.username is None


@pytest.mark.parametrize(
    "candidate",
    [
        "",
        "not-a-url",
        "https://sub.checkout.example.org/cb",
        "https://evil-checkout.example.org/cb",
        "https://checkout.example.org.evil.com/cb",
        "https://evil.com/checkout.example.org",
        "https://checkout.example.org@evil.com/cb",
        "https://evil.com@checkout.example.org/cb",
        "//evil.com",
        "http://checkout.example.org/cb",
        "https://checkout.example.org:444/cb",
        "javascript:https://checkout.example.org",
        "data:text/plain,checkout.example.org",
        "https://checkout.example.org%40evil.com/cb",
        "https://checkout.example.org.evil.com/",
        "lirie://payment-return",
    ],
)
def test_validate_return_url_rejects_lookalikes(monkeypatch, candidate):
    _isolate_checkout_base(monkeypatch)
    with pytest.raises(ValueError, match="return_url"):
        wl.validate_return_url_override(candidate)


def test_require_https_rejects_http_even_on_allowed_host(monkeypatch):
    monkeypatch.delenv("CLIENT_WEB_BASE_URL", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv(
        "WORLDLINE_ALLOWED_RETURN_URL_PREFIXES", "http://checkout.example.org"
    )
    monkeypatch.setenv("WORLDLINE_RETURN_URL_REQUIRE_HTTPS", "1")
    with pytest.raises(ValueError, match="https"):
        wl.validate_return_url_override("http://checkout.example.org/cb")


def test_default_worldline_return_url_uses_client_web_base(monkeypatch):
    monkeypatch.setenv("CLIENT_WEB_BASE_URL", "https://app.example.org")
    result = wl.default_worldline_return_url(42)
    parts = urlsplit(result)
    assert parts.scheme == "https"
    assert parts.hostname == "app.example.org"
    assert parts.path == "/client/payment/worldline/return"
    assert "bookingId=42" in parts.query
