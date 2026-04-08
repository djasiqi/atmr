"""Tests unitaires — noyau pilotage billing (montant observable, qualification basique)."""

from __future__ import annotations

from unittest.mock import MagicMock

from services.admin_booking_billing_kernel import observed_transport_amount, qualify_booking


def test_observed_transport_amount_prefers_positive_amount():
    b = MagicMock()
    b.amount = 42.0
    b.price_amount = 100.0
    assert observed_transport_amount(b) == 42.0


def test_observed_transport_amount_falls_back_to_price_amount():
    b = MagicMock()
    b.amount = 0.0
    b.price_amount = 55.5
    assert observed_transport_amount(b) == 55.5


def test_observed_transport_amount_none_when_both_zero():
    b = MagicMock()
    b.amount = 0.0
    b.price_amount = None
    assert observed_transport_amount(b) is None


def test_qualify_excluded_when_synthetic_demo(monkeypatch):
    b = MagicMock()
    monkeypatch.setattr(
        "services.admin_booking_billing_kernel.is_synthetic_demo_booking",
        lambda _: True,
    )
    out = qualify_booking(b, has_transfer=False, has_pending_transfer=False)
    assert out["state"] == "excluded"
