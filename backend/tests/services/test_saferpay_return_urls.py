"""URLs de retour Saferpay (préfixes autorisés)."""

from __future__ import annotations

import pytest

from services.saferpay import return_urls as ru
from services.saferpay.payment_page import _return_urls_with_outcome_override


def test_validate_return_url_override_ok(monkeypatch):
    monkeypatch.setenv("SAFERPAY_ALLOWED_RETURN_URL_PREFIXES", "https://legacy.example.org")
    u = ru.validate_return_url_override("https://legacy.example.org/cb?x=1")
    assert u.startswith("https://legacy.example.org")


def test_validate_return_url_override_rejects_unknown(monkeypatch):
    monkeypatch.delenv("SAFERPAY_ALLOWED_RETURN_URL_PREFIXES", raising=False)
    monkeypatch.delenv("CLIENT_WEB_BASE_URL", raising=False)
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("SAFERPAY_CHECKOUT_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("WORLDLINE_CHECKOUT_PUBLIC_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="return_url personnalisee interdite"):
        ru.validate_return_url_override("https://evil.com/phish")


def test_validate_return_url_override_allows_app_scheme(monkeypatch):
    monkeypatch.setenv(
        "SAFERPAY_ALLOWED_RETURN_URL_PREFIXES",
        "lirie://payment-return,com.lirie.app://payment-return",
    )
    lirie_return_url = "lirie://payment-return?bookingId=1"
    com_lirie_return_url = "com.lirie.app://payment-return?bookingId=2"
    assert ru.validate_return_url_override(lirie_return_url) == lirie_return_url
    assert ru.validate_return_url_override(com_lirie_return_url) == com_lirie_return_url


def test_validate_return_url_override_rejects_forbidden_scheme(monkeypatch):
    monkeypatch.setenv("SAFERPAY_ALLOWED_RETURN_URL_PREFIXES", "javascript://payment-return")
    with pytest.raises(ValueError, match="schéma interdit"):
        ru.validate_return_url_override("javascript://payment-return?bookingId=1")


def test_default_saferpay_return_urls_prefers_public_base(monkeypatch):
    monkeypatch.delenv("CLIENT_WEB_BASE_URL", raising=False)
    monkeypatch.setenv("SAFERPAY_CHECKOUT_PUBLIC_BASE_URL", "https://tunnel.example.com")
    a, b, c = ru.default_saferpay_return_urls(booking_id=42, payment_id=7)
    assert "bookingId=42" in a
    assert "paymentId=7" in a
    assert a.startswith("https://tunnel.example.com/client/payment/saferpay/return")
    assert b.startswith("https://tunnel.example.com/client/payment/saferpay/return")
    assert c.startswith("https://tunnel.example.com/client/payment/saferpay/return")


def test_default_falls_back_to_public_base_url(monkeypatch):
    monkeypatch.delenv("SAFERPAY_CHECKOUT_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("CLIENT_WEB_BASE_URL", raising=False)
    monkeypatch.delenv("WORLDLINE_CHECKOUT_PUBLIC_BASE_URL", raising=False)
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://app.example.com")
    a, _, _ = ru.default_saferpay_return_urls(booking_id=1, payment_id=2)
    assert a.startswith("https://app.example.com/client/payment/saferpay/return")


def test_default_localhost_blocked_without_flag(monkeypatch):
    monkeypatch.delenv("SAFERPAY_CHECKOUT_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("CLIENT_WEB_BASE_URL", raising=False)
    monkeypatch.delenv("WORLDLINE_CHECKOUT_PUBLIC_BASE_URL", raising=False)
    # Ecraser toute valeur du .env local (ex. SAFERPAY_ALLOW_LOCALHOST_RETURN=1).
    monkeypatch.setenv("SAFERPAY_ALLOW_LOCALHOST_RETURN", "0")
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://localhost:3000")
    with pytest.raises(ValueError, match="localhost"):
        ru.default_saferpay_return_urls(booking_id=1, payment_id=2)


def test_default_localhost_allowed_with_flag(monkeypatch):
    monkeypatch.delenv("SAFERPAY_CHECKOUT_PUBLIC_BASE_URL", raising=False)
    monkeypatch.delenv("CLIENT_WEB_BASE_URL", raising=False)
    monkeypatch.setenv("PUBLIC_BASE_URL", "http://localhost:3000")
    monkeypatch.setenv("SAFERPAY_ALLOW_LOCALHOST_RETURN", "1")
    a, _, _ = ru.default_saferpay_return_urls(booking_id=7, payment_id=3)
    assert "localhost:3000" in a


def test_return_url_override_adds_outcome_and_ids():
    base = "https://app.example.com/client/payment/saferpay/return"
    s, f, a = _return_urls_with_outcome_override(
        base, booking_id=9, payment_id=88
    )
    assert "outcome=success" in s
    assert "outcome=fail" in f
    assert "outcome=abort" in a
    assert "bookingId=9" in s
    assert "paymentId=88" in s


def test_return_url_override_preserves_existing_query():
    base = "https://app.example.com/return?foo=1"
    s, _, _ = _return_urls_with_outcome_override(
        base, booking_id=1, payment_id=2
    )
    assert "foo=1" in s
    assert "outcome=success" in s
