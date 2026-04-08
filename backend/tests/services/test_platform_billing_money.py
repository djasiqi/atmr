from decimal import Decimal

from services.platform_billing.money import money_round_chf


def test_money_round_half_up():
    assert money_round_chf(Decimal("1.005")) == Decimal("1.01")
    assert money_round_chf(Decimal("1.004")) == Decimal("1.00")
    assert money_round_chf(Decimal("10.00")) == Decimal("10.00")
