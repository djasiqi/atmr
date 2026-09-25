"""Résolution e-mail destinataire facture + frais papier Direct patient / PORTAL."""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from application.invoices.paper_invoice_fee import (
    DELIVERY_EMAIL,
    DELIVERY_PAPER,
    PAPER_FEE_DESCRIPTION,
    PAPER_FEE_LINE_META_KEY,
    PAPER_INVOICE_FEE_CHF,
    invoice_has_paper_fee_line,
    normalize_delivery_method,
    resolve_invoice_delivery_method,
)
from services.billing.invoice_recipient_email import resolve_invoice_recipient_email


def test_normalize_delivery_method_defaults_email():
    assert normalize_delivery_method(None) == DELIVERY_EMAIL
    assert normalize_delivery_method("EMAIL") == DELIVERY_EMAIL
    assert normalize_delivery_method("paper") == DELIVERY_PAPER
    assert normalize_delivery_method("courrier") == DELIVERY_PAPER


def test_paper_fee_constant_is_three_chf():
    assert PAPER_INVOICE_FEE_CHF == Decimal("3.00")


def test_invoice_has_paper_fee_line_via_meta():
    line = SimpleNamespace(
        line_meta={PAPER_FEE_LINE_META_KEY: True},
        description="autre",
    )
    inv = SimpleNamespace(lines=[line])
    assert invoice_has_paper_fee_line(inv) is True


def test_invoice_has_paper_fee_line_via_description():
    line = SimpleNamespace(line_meta=None, description=PAPER_FEE_DESCRIPTION)
    inv = SimpleNamespace(lines=[line])
    assert invoice_has_paper_fee_line(inv) is True


def test_invoice_without_paper_fee():
    line = SimpleNamespace(line_meta={}, description="Transport #1")
    inv = SimpleNamespace(lines=[line])
    assert invoice_has_paper_fee_line(inv) is False


def test_resolve_email_from_invoice_meta_first():
    inv = SimpleNamespace(
        meta={"recipient_email": "fige@example.com"},
        client=SimpleNamespace(
            contact_email="contact@example.com",
            user=SimpleNamespace(email="user@example.com"),
        ),
        billing_party=None,
        client_id=None,
        billing_party_id=None,
        lines=[],
    )
    assert resolve_invoice_recipient_email(inv) == "fige@example.com"


def test_resolve_email_from_portal_user_account():
    inv = SimpleNamespace(
        meta={},
        client=SimpleNamespace(
            contact_email=None,
            user=SimpleNamespace(email="osmani.mirjete@gmail.com"),
        ),
        billing_party=SimpleNamespace(contact_email=None),
        client_id=42,
        billing_party_id=None,
        lines=[],
    )
    assert resolve_invoice_recipient_email(inv) == "osmani.mirjete@gmail.com"


def test_resolve_email_from_client_contact_when_no_user():
    inv = SimpleNamespace(
        meta=None,
        client=SimpleNamespace(contact_email="facture@client.ch", user=None),
        billing_party=None,
        client_id=1,
        billing_party_id=None,
        lines=[],
    )
    assert resolve_invoice_recipient_email(inv) == "facture@client.ch"


def test_resolve_email_from_billing_party():
    inv = SimpleNamespace(
        meta={},
        client=SimpleNamespace(contact_email=None, user=None),
        billing_party=SimpleNamespace(contact_email="bp@patient.ch"),
        client_id=1,
        billing_party_id=9,
        lines=[],
    )
    assert resolve_invoice_recipient_email(inv) == "bp@patient.ch"


def test_resolve_email_from_booking_created_debtor_snapshot():
    event = SimpleNamespace(debtor_email_snapshot="snapshot@portal.ch")
    mock_query = MagicMock()
    mock_query.filter_by.return_value.one_or_none.return_value = event

    inv = SimpleNamespace(
        meta={},
        client=SimpleNamespace(contact_email=None, user=None),
        billing_party=SimpleNamespace(contact_email=None),
        client_id=1,
        billing_party_id=None,
        lines=[SimpleNamespace(reservation_id=46759)],
    )

    with patch(
        "models.client_booking_contract_event.ClientBookingContractEvent"
    ) as mock_evt:
        mock_evt.query = mock_query
        assert resolve_invoice_recipient_email(inv) == "snapshot@portal.ch"


def test_resolve_delivery_uses_client_not_explicit():
    client = SimpleNamespace(invoice_delivery_method="email")
    assert (
        resolve_invoice_delivery_method(client=client, explicit="paper")
        == DELIVERY_EMAIL
    )


def test_resolve_delivery_falls_back_to_client():
    client = SimpleNamespace(invoice_delivery_method="paper")
    assert resolve_invoice_delivery_method(client=client) == DELIVERY_PAPER


def test_resolve_delivery_from_invoice_meta_when_no_client():
    inv = SimpleNamespace(meta={"delivery_method": "paper"})
    assert resolve_invoice_delivery_method(invoice=inv) == DELIVERY_PAPER


def test_resolve_email_empty_when_nothing():
    inv = SimpleNamespace(
        meta={},
        client=SimpleNamespace(contact_email=None, user=None),
        billing_party=None,
        client_id=None,
        billing_party_id=None,
        lines=[],
    )
    assert resolve_invoice_recipient_email(inv) is None
