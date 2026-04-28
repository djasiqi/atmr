from decimal import Decimal

from services.saferpay.money import chf_amount_to_saferpay_value_str


def test_chf_to_saferpay_minor_string():
    assert chf_amount_to_saferpay_value_str(45) == "4500"
    assert chf_amount_to_saferpay_value_str(0.5) == "50"


def test_chf_to_saferpay_round_half_up():
    assert chf_amount_to_saferpay_value_str(2.125) == "213"
    assert chf_amount_to_saferpay_value_str(Decimal("12.155")) == "1216"
