"""Tests résolution contrat versionné et fenêtres semi-ouvertes."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from services.platform_billing.contracts import (
    effective_config_for_period,
    month_start_zurich_utc,
    window_contains,
    windows_overlap,
)


def test_window_contains_half_open():
    start = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
    end = datetime(2027, 1, 1, 0, 0, tzinfo=UTC)
    assert window_contains(start, end, datetime(2026, 8, 1, 0, 0, tzinfo=UTC))
    assert window_contains(start, end, datetime(2026, 12, 31, 23, 0, tzinfo=UTC))
    assert not window_contains(start, end, datetime(2027, 1, 1, 0, 0, tzinfo=UTC))


def test_windows_overlap_adjacent_ok():
    a_from = datetime(2026, 8, 1, tzinfo=UTC)
    a_to = datetime(2027, 1, 1, tzinfo=UTC)
    b_from = datetime(2027, 1, 1, tzinfo=UTC)
    b_to = None
    assert not windows_overlap(a_from, a_to, b_from, b_to)


def test_windows_overlap_empty_interval_no_overlap():
    """Fenêtre [A, A) vide : ne chevauche pas [A, ∞)."""
    a = datetime(2026, 8, 1, tzinfo=UTC)
    assert not windows_overlap(a, a, a, None)


def test_windows_overlap_detected():
    a_from = datetime(2026, 8, 1, tzinfo=UTC)
    a_to = datetime(2027, 1, 1, tzinfo=UTC)
    b_from = datetime(2026, 11, 1, tzinfo=UTC)
    b_to = datetime(2027, 2, 1, tzinfo=UTC)
    assert windows_overlap(a_from, a_to, b_from, b_to)


def test_month_start_zurich_utc():
    dt = month_start_zurich_utc(2026, 7)
    assert dt.tzinfo is not None
    # 2026-07-01 00:00 Zurich = 2026-06-30 22:00 UTC (CEST)
    assert dt.year == 2026
    assert dt.month == 6
    assert dt.day == 30


def test_effective_config_for_period_picks_matching_window():
    period_start = month_start_zurich_utc(2026, 7)
    cfg_old = MagicMock()
    cfg_old.id = 1
    cfg_old.effective_from = datetime(2026, 1, 1, tzinfo=UTC)
    cfg_old.effective_to = datetime(2026, 7, 1, tzinfo=UTC)
    cfg_new = MagicMock()
    cfg_new.id = 2
    cfg_new.effective_from = datetime(2026, 7, 1, tzinfo=UTC)
    cfg_new.effective_to = None

    mock_query = MagicMock()
    mock_query.filter.return_value.order_by.return_value.all.return_value = [
        cfg_new,
        cfg_old,
    ]
    with patch(
        "services.platform_billing.contracts.CompanyPlatformBillingConfig"
    ) as M:
        M.query = mock_query
        # July 1 Zurich is still before cfg_old.effective_to if effective_to is July 1 UTC
        # period_start for July is June 30 22:00 UTC — contained in cfg_old [Jan1, Jul1)
        # and also in cfg_new if from is Jul1 UTC... June 30 < Jul1 so only cfg_old
        chosen = effective_config_for_period(10, period_start)
        assert chosen is cfg_old


def test_commissionable_amount_prefers_price_amount():
    from decimal import Decimal

    from services.platform_billing.commissionable_amount import (
        AmountConfidence,
        CommissionAmountSource,
        resolve_commissionable_amount,
    )

    b = MagicMock()
    b.status = "COMPLETED"
    b.price_amount = Decimal("100.00")
    b.amount = 50.0
    b.cancellation_fee_amount = None
    b.final_billable_amount = None
    b.locked_price_amount = None
    r = resolve_commissionable_amount(b)
    assert r.amount == Decimal("100.00")
    assert r.source == CommissionAmountSource.PRICE_AMOUNT
    assert r.confidence == AmountConfidence.CERTAIN


def test_platform_qr_amount_no_5_cents():
    from decimal import Decimal

    from services.platform_billing.swiss_qr import platform_qr_amount

    assert platform_qr_amount(Decimal("336.84")) == Decimal("336.84")
