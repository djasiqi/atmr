"""Table-driven tests for compute_cancellation_fee()."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest

from application.bookings.cancellation_rules import (
    CancellationFeeResult,
    compute_cancellation_fee,
)

SAMPLE_POLICY = {
    "enabled": True,
    "basis": "booking_amount",
    "apply_when_driver_assigned_only": True,
    "tiers": [
        {"id": "t1", "type": "time", "hours_before": 2, "percent": 60, "label": "< 2h"},
        {"id": "t2", "type": "time", "hours_before": 12, "percent": 40, "label": "< 12h"},
        {"id": "t3", "type": "time", "hours_before": 24, "percent": 20, "label": "< 24h"},
        {"id": "t4", "type": "status", "status": "EN_ROUTE", "percent": 70, "label": "Chauffeur en route"},
    ],
    "min_fee_chf": 0,
    "max_fee_chf": None,
    "reason_overrides": {
        "MAJOR_DELAY": {"billable": False},
        "NO_SHOW": {"billable": True},
    },
}


def _booking(amount=100, driver_id=1, scheduled_hours_from_now=10, status="ASSIGNED"):
    sched = datetime.now(UTC) + timedelta(hours=scheduled_hours_from_now)
    return SimpleNamespace(
        amount=amount,
        driver_id=driver_id,
        scheduled_time=sched,
        status=status,
        company_id=1,
    )


def _cancel_at(booking, hours_before):
    return booking.scheduled_time - timedelta(hours=hours_before)


class TestComputeCancellationFee:
    """Table-driven tests for the fee computation algorithm."""

    def test_policy_none_legacy_billable(self):
        """policy=None -> legacy reason-based, fee=0."""
        b = _booking()
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=datetime.now(UTC), reason_code="LAST_MINUTE",
            policy=None,
        )
        assert r.is_billable is True
        assert r.fee_amount == Decimal("0")
        assert r.tier_id is None

    def test_policy_none_legacy_non_billable(self):
        """policy=None, non-billable reason -> is_billable=False."""
        b = _booking()
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=datetime.now(UTC), reason_code="COMPANY_ISSUE",
            policy=None,
        )
        assert r.is_billable is False
        assert r.fee_amount == Decimal("0")

    def test_policy_disabled_legacy(self):
        """policy.enabled=false -> same as None."""
        b = _booking()
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=datetime.now(UTC), reason_code="LAST_MINUTE",
            policy={"enabled": False, "tiers": []},
        )
        assert r.is_billable is True
        assert r.fee_amount == Decimal("0")
        assert r.tier_id is None

    def test_no_driver_apply_when_driver_assigned(self):
        """No driver + apply_when_driver_assigned_only -> is_billable=False, fee=0."""
        b = _booking(driver_id=None)
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 10), reason_code="LAST_MINUTE",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is False
        assert r.fee_amount == Decimal("0")

    def test_amount_none(self):
        """booking.amount=None -> is_billable=False, fee=0."""
        b = _booking(amount=None)
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 10), reason_code="LAST_MINUTE",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is False
        assert r.fee_amount == Decimal("0")

    def test_amount_zero(self):
        """booking.amount=0 -> is_billable=False."""
        b = _booking(amount=0)
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 10), reason_code="LAST_MINUTE",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is False

    def test_cancel_30h_no_tier_match(self):
        """Cancel 30h before -> no tier matches -> fee=0, is_billable=False."""
        b = _booking(scheduled_hours_from_now=30)
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 30), reason_code="LAST_MINUTE",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is False
        assert r.fee_amount == Decimal("0")
        assert r.tier_id is None

    def test_cancel_20h_tier_24h(self):
        """Cancel 20h before -> tier 24h (20%) -> fee = 100 * 0.20 = 20."""
        b = _booking(amount=100, scheduled_hours_from_now=20)
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 20), reason_code="LAST_MINUTE",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is True
        assert r.percent == 20
        assert r.tier_id == "t3"
        assert r.fee_amount == Decimal("20.00")

    def test_cancel_10h_tier_12h(self):
        """Cancel 10h before -> tier 12h (40%) -> fee = 100 * 0.40 = 40."""
        b = _booking(amount=100, scheduled_hours_from_now=10)
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 10), reason_code="CLIENT_REQUEST",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is True
        assert r.percent == 40
        assert r.tier_id == "t2"
        assert r.fee_amount == Decimal("40.00")

    def test_cancel_1h_tier_2h(self):
        """Cancel 1h before -> tier 2h (60%) -> fee = 100 * 0.60 = 60."""
        b = _booking(amount=100, scheduled_hours_from_now=5)
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 1), reason_code="LAST_MINUTE",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is True
        assert r.percent == 60
        assert r.tier_id == "t1"
        assert r.fee_amount == Decimal("60.00")

    def test_cancel_after_scheduled_clamp_zero(self):
        """Cancel after scheduled_time -> hours_before clamped to 0 -> tier 2h (most strict)."""
        b = _booking(amount=100, scheduled_hours_from_now=-1)
        cancel_time = b.scheduled_time + timedelta(hours=1)
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=cancel_time, reason_code="LAST_MINUTE",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is True
        assert r.tier_id == "t1"
        assert r.percent == 60

    def test_cancel_en_route_status_tier(self):
        """EN_ROUTE -> status tier (70%) -> fee = 100 * 0.70 = 70."""
        b = _booking(amount=100, status="EN_ROUTE")
        r = compute_cancellation_fee(
            b, status_at_cancel="EN_ROUTE",
            cancelled_at=datetime.now(UTC), reason_code="LAST_MINUTE",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is True
        assert r.percent == 70
        assert r.tier_id == "t4"
        assert r.fee_amount == Decimal("70.00")

    def test_min_fee_clamp(self):
        """Fee below min -> clamped to min."""
        policy = {**SAMPLE_POLICY, "min_fee_chf": 30}
        b = _booking(amount=100, scheduled_hours_from_now=20)
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 20), reason_code="LAST_MINUTE",
            policy=policy,
        )
        assert r.fee_amount == Decimal("30.00")

    def test_max_fee_clamp(self):
        """Fee above max -> clamped to max."""
        policy = {**SAMPLE_POLICY, "max_fee_chf": 25}
        b = _booking(amount=100, scheduled_hours_from_now=10)
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 10), reason_code="LAST_MINUTE",
            policy=policy,
        )
        assert r.fee_amount == Decimal("25.00")

    def test_reason_override_non_billable(self):
        """MAJOR_DELAY override -> non-billable, fee=0."""
        b = _booking()
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 5), reason_code="MAJOR_DELAY",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is False
        assert r.fee_amount == Decimal("0")

    def test_cascade_from_outbound(self):
        """Cascade -> non-billable, fee=0 regardless of policy."""
        b = _booking()
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=datetime.now(UTC), reason_code="LAST_MINUTE",
            cancel_source="cascade_from_outbound",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is False
        assert r.fee_amount == Decimal("0")

    def test_non_billable_reason_no_override(self):
        """COMPANY_ISSUE (not in BILLABLE_REASONS, no override) -> non-billable."""
        b = _booking()
        r = compute_cancellation_fee(
            b, status_at_cancel="ASSIGNED",
            cancelled_at=_cancel_at(b, 5), reason_code="COMPANY_ISSUE",
            policy=SAMPLE_POLICY,
        )
        assert r.is_billable is False
        assert r.fee_amount == Decimal("0")
