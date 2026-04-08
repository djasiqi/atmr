"""Tests moteur facturation plateforme : verrouillage, arrondi agrégat commission."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from models.enums import PlatformBillingPeriodStatus
from services.platform_billing.engine import recalculate_platform_period_drafts
from services.platform_billing.money import money_round_chf


def test_recalculate_raises_when_period_locked():
    period = MagicMock()
    period.status = PlatformBillingPeriodStatus.LOCKED.value
    with patch("services.platform_billing.engine.db.session.get", return_value=period):
        with pytest.raises(ValueError, match="verrouill"):
            recalculate_platform_period_drafts(42)


def test_commission_total_is_sum_of_per_booking_rounded_amounts():
    """§2.9 : arrondi par booking puis somme (pas d’arrondi sur le total)."""
    rate = Decimal("0.075")
    amounts = [Decimal("10.333"), Decimal("20.666"), Decimal("5.111")]
    total = sum(money_round_chf(a * rate) for a in amounts)
    assert total == money_round_chf(total)
    assert total == Decimal("2.70")


def test_money_round_chf_half_up_edge_cases():
    assert money_round_chf(Decimal("1.005")) == Decimal("1.01")
    assert money_round_chf(Decimal("1.004")) == Decimal("1.00")
