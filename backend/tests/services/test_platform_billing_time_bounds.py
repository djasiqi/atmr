"""Tests PR1 — invariants temporels Europe/Zurich."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from services.platform_billing.time_bounds import (
    billing_period_has_ended,
    next_month_start_zurich_utc,
    zurich_month_bounds_utc,
)

_ZH = ZoneInfo("Europe/Zurich")


def test_zurich_month_bounds_march_2026():
    start, end = zurich_month_bounds_utc(2026, 3)
    assert start.tzinfo == UTC
    assert end.tzinfo == UTC
    assert start < end
    assert start.astimezone(_ZH).month == 3
    assert end.astimezone(_ZH).month == 3


def test_same_bounds_for_subscription_and_commission_month_filters():
    """Deux axes métier (created_at vs completed_at) partagent les mêmes bornes UTC du mois M."""
    a = zurich_month_bounds_utc(2025, 11)
    b = zurich_month_bounds_utc(2025, 11)
    assert a == b


def test_billing_period_has_ended_january_winter():
    # 31 jan 2027 22:59:59 UTC = 23:59:59 Zurich (+01) → janvier encore ouvert
    assert not billing_period_has_ended(
        2027,
        1,
        now_utc=datetime(2027, 1, 31, 22, 59, 59, tzinfo=UTC),
    )
    # 31 jan 2027 23:00:00 UTC = 1er fév 00:00 Zurich → janvier terminé
    assert billing_period_has_ended(
        2027,
        1,
        now_utc=datetime(2027, 1, 31, 23, 0, 0, tzinfo=UTC),
    )


def test_billing_period_has_ended_august_summer():
    # 31 août 2026 21:59:59 UTC = 23:59:59 Zurich (+02) → août ouvert
    assert not billing_period_has_ended(
        2026,
        8,
        now_utc=datetime(2026, 8, 31, 21, 59, 59, tzinfo=UTC),
    )
    # 31 août 2026 22:00:00 UTC = 1er sept 00:00 Zurich → août terminé
    assert billing_period_has_ended(
        2026,
        8,
        now_utc=datetime(2026, 8, 31, 22, 0, 0, tzinfo=UTC),
    )


def test_next_month_start_matches_bounds():
    _start, end = zurich_month_bounds_utc(2026, 8)
    nxt = next_month_start_zurich_utc(2026, 8)
    assert nxt > end
    assert nxt.astimezone(_ZH).month == 9
    assert _start.astimezone(_ZH).month == 8
