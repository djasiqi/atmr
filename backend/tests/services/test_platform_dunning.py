"""Tests P0 — dunning art. 6 bis (capabilities, overdue, policy freeze)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from models.enums import (
    LegalForm,
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
from services.platform_billing.partner_agreement_versions import (
    PACK_SCHEMA_VERSION,
    PARTICULAR_VERSION,
)


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


def test_set_billing_access_admin_priority():
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

    with (
        patch(
            "services.platform_billing.capabilities.db.session.get",
            return_value=company,
        ),
        patch("services.platform_billing.capabilities.db.session.flush"),
    ):
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


def test_particular_mentions_progressive_suspension():
    from io import BytesIO

    from pypdf import PdfReader

    from services.platform_billing.partner_agreement_canonical import (
        ensure_canonical_documents,
    )
    from services.platform_billing.partner_agreement_particular_content import (
        build_particular_agreement_content,
    )
    from services.platform_billing.partner_agreement_particular_pdf import (
        build_particular_pdf_bytes,
    )

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
            "signatory_name": "Drin Jasiqi",
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
        "lirie_commission_enabled": True,
        "own_portfolio_billing_enabled": True,
        "commission_rate": "0.100000",
        "payment_terms_days": 30,
        "statement_dispute_days": 10,
        "free_license_max_months": 12,
    }
    canon = ensure_canonical_documents()
    content = build_particular_agreement_content(
        reference="LIRIE/PART/2026-08/002",
        parties=parties,
        commercial=commercial,
        agreement_effective_from="2026-08-01",
        general_terms_sha256=canon["general_terms"].sha256,
        dpa_sha256=canon["dpa"].sha256,
    )
    pdf = build_particular_pdf_bytes(content)
    text = "\n".join((p.extract_text() or "") for p in PdfReader(BytesIO(pdf)).pages)
    assert PACK_SCHEMA_VERSION == "lirie-partner-pack-v1"
    assert PARTICULAR_VERSION in text
    assert "suspension" in text.lower()
    assert "impay" in text.lower()


def test_canonical_terms_cover_dunning_procedure():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "assets"
        / "contracts"
        / "canonical"
        / "sources"
        / "lirie-partner-terms-v1.20.md"
    )
    md = source.read_text(encoding="utf-8")
    assert "Procédure de rappel" in md or "suspension" in md.lower()
    assert "défaut de paiement" in md.lower()
