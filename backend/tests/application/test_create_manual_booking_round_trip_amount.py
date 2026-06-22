"""Répartition tarifaire aller-retour — création manuelle."""

import pytest

from application.companies.reservations.create_manual_booking import (
    CreateManualBookingError,
    _preferential_rate_to_booking_amount,
    _resolve_leg_amounts,
    _validate_round_trip_leg_amounts,
)


def test_preferential_rate_doubles_for_round_trip():
    assert _preferential_rate_to_booking_amount(35.0, is_round_trip=True) == 70.0
    assert _preferential_rate_to_booking_amount(35.0, is_round_trip=False) == 35.0


def test_preferential_round_trip_assigns_per_leg_amount():
    out_amt, ret_amt, out_price, ret_price = _resolve_leg_amounts(
        70.0,
        is_round_trip=True,
        price_total=70.0,
        preferential_per_leg=35.0,
    )
    assert out_amt == 35.0
    assert ret_amt == 35.0
    assert out_price == 35.0
    assert ret_price == 35.0
    _validate_round_trip_leg_amounts(out_amt, ret_amt, 70.0)


def test_resolve_leg_amounts_splits_simulated_round_trip_total():
    out_amt, ret_amt, out_price, ret_price = _resolve_leg_amounts(
        70.0,
        is_round_trip=True,
        price_total=70.0,
    )
    assert out_amt == 35.0
    assert ret_amt == 35.0
    assert out_price == 35.0
    assert ret_price == 35.0
    assert out_amt + ret_amt == 70.0


def test_resolve_leg_amounts_one_way_keeps_return_zero():
    out_amt, ret_amt, out_price, ret_price = _resolve_leg_amounts(
        35.0,
        is_round_trip=False,
        price_total=None,
    )
    assert out_amt == 35.0
    assert ret_amt == 0.0
    assert out_price == 35.0
    assert ret_price is None


@pytest.mark.parametrize(
    ("outbound", "return_leg", "total"),
    [
        (70.0, 0.0, 70.0),
        (0.0, 70.0, 70.0),
        (70.0, 70.0, 70.0),
        (50.0, 10.0, 70.0),
    ],
)
def test_validate_rejects_invalid_round_trip_splits(
    outbound: float, return_leg: float, total: float
):
    with pytest.raises(CreateManualBookingError):
        _validate_round_trip_leg_amounts(outbound, return_leg, total)
