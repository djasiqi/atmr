"""Tests P0 — dunning art. 6 bis (capabilities, overdue, policy freeze)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.enums import (
    PlatformBillingAccessState,
    PlatformBillingStateSource,
    PlatformDunningCaseStatus,
    PlatformDunningEventStatus,
    PlatformDunningEventType,
    PlatformIssuedInvoiceStatus,
)
from services.platform_billing.capabilities import (
    BillingCapability,
    is_billing_capability_allowed,
    set_billing_access_state,
)
from services.platform_billing.dunning import (
    enforceable_balance,
    is_invoice_overdue_enforceable,
)
from services.platform_billing.dunning_policy import (
    build_dunning_policy_snapshot,
    compute_dunning_automation_ready,
    parse_dunning_fields,
)
from services.platform_billing.partner_agreement_docx import (
    TEMPLATE_VERSION,
    build_partner_agreement_docx_bytes,
)
from models.enums import LegalForm


def test_parse_dunning_invariant_full_after_grace():
    with pytest.raises(ValueError, match="full_suspend"):
        parse_dunning_fields(
            {
                "reminder_delay_days_after_due": 10,
                "reminder_grace_days": 20,
                "full_suspend_days_after_due": 25,
            }
        )


def test_dunning_automation_ready_requires_signed():
    cfg = SimpleNamespace(
        id=1,
        automated_dunning_enabled=True,
    )
    ready = compute_dunning_automation_ready(cfg=cfg, agreement=None)
    assert ready["ready"] is False
    assert "no_partner_agreement" in ready["reasons"]


def test_policy_snapshot_version():
    cfg = SimpleNamespace(**parse_dunning_fields({}))
    snap = build_dunning_policy_snapshot(cfg)
    assert snap["policy_version"] == 1
    assert snap["reminder_grace_days"] == 10
    assert snap["full_suspend_days_after_due"] == 30


def test_overdue_requires_sent_and_balance():
    now = datetime(2026, 8, 1, tzinfo=UTC)
    inv = SimpleNamespace(
        status=PlatformIssuedInvoiceStatus.ISSUED.value,
        sent_at=None,
        due_at=now - timedelta(days=1),
        total_amount=Decimal("100"),
        amount_paid=Decimal("0"),
        id=1,
    )
    with patch(
        "services.platform_billing.dunning.disputed_amount_active",
        return_value=Decimal("0"),
    ):
        assert is_invoice_overdue_enforceable(inv, now=now) is False
        inv.sent_at = now - timedelta(days=2)
        inv.status = PlatformIssuedInvoiceStatus.SENT.value
        assert is_invoice_overdue_enforceable(inv, now=now) is True
        inv.amount_paid = Decimal("100")
        assert is_invoice_overdue_enforceable(inv, now=now) is False


def test_enforceable_balance_minus_dispute():
    inv = SimpleNamespace(total_amount=Decimal("100"), amount_paid=Decimal("10"), id=1)
    with patch(
        "services.platform_billing.dunning.disputed_amount_active",
        return_value=Decimal("20"),
    ):
        assert enforceable_balance(inv) == Decimal("70.00")


def test_set_billing_access_admin_priority(app_ctx=None):
    """admin_manual ne doit pas être écrasé par automatic_dunning."""
    company = MagicMock()
    company.platform_billing_access_state = PlatformBillingAccessState.FULL.value
    company.platform_billing_state_source = (
        PlatformBillingStateSource.ADMIN_MANUAL.value
    )
    company.platform_billing_state_since = datetime.now(UTC)
    company.platform_billing_state_reason_code = "admin"
    company.platform_billing_state_config_id = None
    company.platform_billing_state_updated_at = None

    with patch(
        "services.platform_billing.capabilities.db.session.get",
        return_value=company,
    ):
        with patch("services.platform_billing.capabilities.db.session.flush"):
            result = set_billing_access_state(
                1,
                PlatformBillingAccessState.ACTIVE.value,
                source=PlatformBillingStateSource.AUTOMATIC_DUNNING.value,
                reason_code="auto",
            )
            assert (
                result.platform_billing_access_state
                == PlatformBillingAccessState.FULL.value
            )


def test_docx_v14_contains_configurable_dunning():
    import io
    import zipfile

    parties = {
        "operator": {
            "legal_name": "Drin Jasiqi",
            "legal_form": LegalForm.SOLE_PROPRIETORSHIP.value,
            "legal_form_label": "Indépendant",
            "street_name": "Avenue Ernest-Pictet",
            "building_number": "9",
            "postal_code": "1203",
            "city": "Genève",
            "country_code": "CH",
            "uid_ide": None,
            "signatory_name": "Drin Jasiqi",
            "signatory_title": "Exploitant",
            "contractual_email": "info@lirie.ch",
        },
        "partner": {
            "legal_name": "Emmenez-moi Sàrl",
            "legal_form": LegalForm.SARL.value,
            "legal_form_label": "Sàrl",
            "street_name": "Route de Chevrens",
            "building_number": "145",
            "postal_code": "1247",
            "city": "Anières",
            "country_code": "CH",
            "uid_ide": "CHE-273.048.653",
            "signatory_name": "Khalid ALAOUI",
            "signatory_title": "associé-gérant",
            "contractual_email": "a@b.ch",
        },
    }
    commercial = {
        "subscription_pricing_mode": "free",
        "free_license_max_months": 60,
        "lirie_commission_enabled": True,
        "own_portfolio_billing_enabled": True,
        "commission_rate": "0.100000",
        "commission_cancellation_policy": "exclude",
        "payment_terms_days": 30,
        "statement_dispute_days": 10,
        "automated_dunning_enabled": True,
        "reminder_delay_days_after_due": 0,
        "reminder_grace_days": 10,
        "full_suspend_days_after_due": 30,
        "full_suspend_overdue_invoice_count": 2,
        "termination_notice_days": 10,
        "partial_block_marketplace_offers": True,
        "partial_block_marketplace_acceptance": True,
        "partial_block_billable_support": True,
        "partial_block_billable_configuration": True,
    }
    raw = build_partner_agreement_docx_bytes(
        reference="LIRIE/PART/2026-08/002",
        parties=parties,
        commercial=commercial,
        agreement_effective_from="2026-08-01",
    )
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        xml = zf.read("word/document.xml").decode("utf-8")
    assert TEMPLATE_VERSION == "lirie-partner-v1.4"
    assert "lirie-partner-v1.4" in xml
    assert "factures échues et impayées" in xml
    assert "10 jours calendaires" in xml or "10" in xml


def test_docx_automation_off_no_renunciation():
    import io
    import zipfile

    parties = {
        "operator": {
            "legal_name": "Drin Jasiqi",
            "legal_form": LegalForm.SOLE_PROPRIETORSHIP.value,
            "street_name": "Avenue Ernest-Pictet",
            "building_number": "9",
            "postal_code": "1203",
            "city": "Genève",
            "country_code": "CH",
            "signatory_name": "Drin Jasiqi",
        },
        "partner": {
            "legal_name": "Emmenez-moi Sàrl",
            "legal_form": LegalForm.SARL.value,
            "legal_form_label": "Sàrl",
            "street_name": "Route",
            "building_number": "1",
            "postal_code": "1200",
            "city": "Genève",
            "country_code": "CH",
            "uid_ide": "CHE-1",
            "signatory_name": "X",
        },
    }
    commercial = {
        "automated_dunning_enabled": False,
        "lirie_commission_enabled": False,
        "own_portfolio_billing_enabled": False,
    }
    raw = build_partner_agreement_docx_bytes(
        reference="LIRIE/PART/2026-08/003",
        parties=parties,
        commercial=commercial,
        agreement_effective_from="2026-08-01",
    )
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        xml = zf.read("word/document.xml").decode("utf-8")
    assert "mesures automatisées" in xml
    assert "conserve ses droits" in xml
