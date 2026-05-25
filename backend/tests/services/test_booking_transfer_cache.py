"""Tests du cache batch transferts pour listes réservations."""

from unittest.mock import MagicMock, patch

from services.companies.booking_transfer_cache import (
    attach_transfer_cache_to_bookings,
    build_transfer_cache_for_bookings,
)


def _booking(bid, company_id=1, executing_company_id=None):
    b = MagicMock()
    b.id = bid
    b.company_id = company_id
    b.executing_company_id = executing_company_id
    return b


@patch("services.companies.booking_transfer_cache.BookingTransfer")
def test_build_transfer_cache_empty(mock_transfer_model):
    assert build_transfer_cache_for_bookings([]) == {}
    mock_transfer_model.query.filter.assert_not_called()


@patch("services.companies.booking_transfer_cache.BookingTransfer")
def test_attach_transfer_cache_sets_attribute(mock_transfer_model):
    mock_transfer_model.query.filter.return_value.filter.return_value.all.return_value = []
    bookings = [_booking(1), _booking(2)]
    attach_transfer_cache_to_bookings(bookings)
    assert hasattr(bookings[0], "_transfer_cache")
    assert bookings[0]._transfer_cache["is_transferred"] is False
