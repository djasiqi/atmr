"""Tests PR1 — workflow validate / lock / issue et refus d'invariants."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from models.enums import (
    PlatformBillingPeriodStatus,
    PlatformStatementStatus,
)
from services.platform_billing.engine import (
    assert_billing_period_has_ended,
    lock_platform_billing_period,
    validate_statement,
)
from services.platform_billing.errors import BillingInvariantError
from services.platform_billing.issuance import issue_platform_invoice
from services.platform_billing.money import money_round_chf


def test_recalculate_raises_when_period_locked():
    from services.platform_billing.engine import recalculate_platform_period_drafts

    period = MagicMock()
    period.status = PlatformBillingPeriodStatus.LOCKED.value
    with (
        patch("services.platform_billing.engine.db.session.get", return_value=period),
        pytest.raises(ValueError, match="verrouill"),
    ):
        recalculate_platform_period_drafts(42)


def test_commission_total_is_sum_of_per_booking_rounded_amounts():
    rate = Decimal("0.075")
    amounts = [Decimal("10.333"), Decimal("20.666"), Decimal("5.111")]
    total = sum(money_round_chf(a * rate) for a in amounts)
    assert total == money_round_chf(total)
    assert total == Decimal("2.70")


def test_money_round_chf_half_up_edge_cases():
    # Arrondi plateforme = multiples de 0,05 CHF
    assert money_round_chf(Decimal("1.005")) == Decimal("1.00")
    assert money_round_chf(Decimal("1.004")) == Decimal("1.00")
    assert money_round_chf(Decimal("1.025")) == Decimal("1.05")
    assert money_round_chf(Decimal("1687.14")) == Decimal("1687.15")


def test_assert_period_still_open_raises():
    with pytest.raises(BillingInvariantError) as ei:
        assert_billing_period_has_ended(
            2026,
            8,
            now_utc=datetime(2026, 8, 15, 12, 0, 0, tzinfo=UTC),
        )
    assert ei.value.code == "PERIOD_STILL_OPEN"


def test_validate_refuses_needs_review():
    inv = MagicMock()
    inv.statement_status = PlatformStatementStatus.NEEDS_REVIEW.value
    period = MagicMock()
    period.billing_year = 2026
    period.billing_month = 7
    inv.period = period
    with (
        patch("services.platform_billing.engine.db.session.get", return_value=inv),
        patch("services.platform_billing.engine.assert_billing_period_has_ended"),
        pytest.raises(BillingInvariantError) as ei,
    ):
        validate_statement(1, now_utc=datetime(2026, 8, 2, tzinfo=UTC))
    assert ei.value.code == "STATEMENT_REVIEW_REQUIRED"


def test_validate_refuses_draft():
    inv = MagicMock()
    inv.statement_status = PlatformStatementStatus.DRAFT.value
    period = MagicMock()
    period.billing_year = 2026
    period.billing_month = 7
    inv.period = period
    with (
        patch("services.platform_billing.engine.db.session.get", return_value=inv),
        patch("services.platform_billing.engine.assert_billing_period_has_ended"),
        pytest.raises(BillingInvariantError) as ei,
    ):
        validate_statement(1)
    assert ei.value.code == "INVALID_STATEMENT_TRANSITION"


def test_validate_accepts_calculated():
    inv = MagicMock()
    inv.statement_status = PlatformStatementStatus.CALCULATED.value
    period = MagicMock()
    period.billing_year = 2026
    period.billing_month = 7
    inv.period = period
    with (
        patch("services.platform_billing.engine.db.session.get", return_value=inv),
        patch("services.platform_billing.engine.assert_billing_period_has_ended"),
        patch("services.platform_billing.engine.db.session.commit"),
    ):
        out = validate_statement(1)
    assert out.statement_status == PlatformStatementStatus.VALIDATED.value


def test_lock_refuses_when_readiness_not_ready():
    readiness = {
        "ready_to_lock": False,
        "blocking_reasons": [
            {
                "code": "STATEMENTS_NOT_VALIDATED",
                "message": "Tous les relevés doivent être VALIDATED avant clôture.",
                "company_ids": [3],
            }
        ],
    }
    with (
        patch(
            "services.platform_billing.engine.build_platform_billing_period_readiness",
            return_value=readiness,
        ),
        pytest.raises(BillingInvariantError) as ei,
    ):
        lock_platform_billing_period(9)
    assert ei.value.code == "STATEMENTS_NOT_VALIDATED"


def test_issue_refuses_validated_without_lock():
    statement = MagicMock()
    statement.statement_status = PlatformStatementStatus.VALIDATED.value
    period = MagicMock()
    period.billing_year = 2026
    period.billing_month = 7
    period.status = PlatformBillingPeriodStatus.DRAFT.value
    period.id = 1
    statement.period = period
    statement.company_id = 5

    mock_q = MagicMock()
    mock_q.filter_by.return_value.first.return_value = None
    with (
        patch(
            "services.platform_billing.issuance.db.session.get",
            return_value=statement,
        ),
        patch("services.platform_billing.issuance.PlatformIssuedInvoice") as Issued,
        patch("services.platform_billing.engine.assert_billing_period_has_ended"),
        pytest.raises(BillingInvariantError) as ei,
    ):
        Issued.query = mock_q
        issue_platform_invoice(12)
    assert ei.value.code == "PERIOD_NOT_LOCKED"


def test_issue_refuses_validated_even_if_period_locked():
    """Plus de promotion silencieuse VALIDATED → LOCKED."""
    statement = MagicMock()
    statement.statement_status = PlatformStatementStatus.VALIDATED.value
    period = MagicMock()
    period.billing_year = 2026
    period.billing_month = 7
    period.status = PlatformBillingPeriodStatus.LOCKED.value
    period.id = 1
    statement.period = period

    mock_q = MagicMock()
    mock_q.filter_by.return_value.first.return_value = None
    with (
        patch(
            "services.platform_billing.issuance.db.session.get",
            return_value=statement,
        ),
        patch("services.platform_billing.issuance.PlatformIssuedInvoice") as Issued,
        patch("services.platform_billing.engine.assert_billing_period_has_ended"),
        pytest.raises(BillingInvariantError) as ei,
    ):
        Issued.query = mock_q
        issue_platform_invoice(12)
    assert ei.value.code == "STATEMENT_NOT_LOCKED"
