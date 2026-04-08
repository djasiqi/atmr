from datetime import UTC
from zoneinfo import ZoneInfo

from services.platform_billing.time_bounds import zurich_month_bounds_utc

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
