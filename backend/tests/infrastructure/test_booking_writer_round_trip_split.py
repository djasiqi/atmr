"""Tests unitaires : répartition 50/50 du tarif aller-retour à la persistance."""

from infrastructure.persistence.bookings.booking_writer import (
    _split_round_trip_total_amount,
)


def test_split_round_trip_90_chf():
    a, b = _split_round_trip_total_amount(90.0)
    assert a == 45.0
    assert b == 45.0
    assert a + b == 90.0


def test_split_round_trip_odd_cent_sum_exact():
    a, b = _split_round_trip_total_amount(10.01)
    assert round(a + b, 2) == 10.01


def test_split_below_two_min_legs_all_on_outbound():
    a, b = _split_round_trip_total_amount(0.8)
    assert a == 0.8
    assert b == 0.0
