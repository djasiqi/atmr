from decimal import Decimal

from services.platform_billing.money import money_round_chf


def test_money_round_chf_five_cents():
    assert money_round_chf(Decimal("1687.14")) == Decimal("1687.15")
    assert money_round_chf(Decimal("10.32")) == Decimal("10.30")
    assert money_round_chf(Decimal("10.33")) == Decimal("10.35")
    assert money_round_chf(Decimal("10.36")) == Decimal("10.35")
    assert money_round_chf(Decimal("10.38")) == Decimal("10.40")
    assert money_round_chf(Decimal("10.00")) == Decimal("10.00")


def test_money_round_chf_half_up_edge():
    # 1.025 → exactement à mi-chemin de 1.00 et 1.05 → HALF_UP → 1.05
    assert money_round_chf(Decimal("1.025")) == Decimal("1.05")
    assert money_round_chf(Decimal("1.024")) == Decimal("1.00")
    assert money_round_chf(Decimal("1.005")) == Decimal("1.00")
    assert money_round_chf(Decimal("1.004")) == Decimal("1.00")
