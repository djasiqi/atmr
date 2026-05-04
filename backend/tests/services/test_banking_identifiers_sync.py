"""Tests pour la synchronisation IBAN / QR-IBAN multi-modèles."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_sync_billing_settings_source_updates_company_and_profile():
    """Source billing_settings : company + profil QR alignés sur settings."""
    billing = MagicMock()
    billing.iban = "CH1500788000051291632"
    billing.qr_iban = None

    profile = MagicMock()

    company = SimpleNamespace(id=1)

    with (
        patch("models.CompanyBillingSettings") as Cbs,
        patch("models.CompanyBillingProfile") as Cbp,
    ):
        Cbs.query.filter_by.return_value.first.return_value = billing
        Cbp.query.filter_by.return_value.first.return_value = profile

        from services.billing.banking_identifiers_sync import sync_banking_identifiers

        sync_banking_identifiers(company, source="billing_settings")

    assert company.iban == "CH1500788000051291632"
    assert profile.iban == "CH1500788000051291632"
    assert profile.qr_iban == "CH1500788000051291632"


def test_sync_company_source_creates_billing_and_updates_profile():
    """Source company : propage vers CompanyBillingSettings + profil."""
    company = SimpleNamespace(id=2, iban="CH93 0076 2011 6238 5295 7")

    profile = MagicMock()

    mock_session = MagicMock()
    with (
        patch("models.CompanyBillingSettings") as Cbs,
        patch("models.CompanyBillingProfile") as Cbp,
        patch("ext.db") as db_mod,
    ):
        Cbs.query.filter_by.return_value.first.return_value = None
        Cbp.query.filter_by.return_value.first.return_value = profile
        db_mod.session = mock_session

        from services.billing.banking_identifiers_sync import sync_banking_identifiers

        sync_banking_identifiers(company, source="company")

    assert mock_session.add.called
    new_billing = mock_session.add.call_args[0][0]
    assert new_billing.company_id == 2
    assert new_billing.iban == "CH9300762011623852957"
    assert profile.iban == "CH9300762011623852957"
