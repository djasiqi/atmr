"""Tests pour l'agrégat Booking."""

from __future__ import annotations

import pytest

from bookings.domain.booking import Booking
from bookings.domain.booking_id import BookingId
from bookings.domain.value_objects import Amount, BookingStatus, Location

# Constantes pour les tests
TEST_AMOUNT = 25.0
TEST_DRIVER_ID = 10


class TestBookingAggregate:
    """Tests pour l'agrégat Booking."""

    def test_create_booking(self):
        """Test création d'un booking."""
        booking = Booking(
            id=BookingId(1),
            company_id=1,
            client_id=1,
            user_id=1,
            customer_name="Test Customer",
            pickup_location=Location(
                address="123 Main St", latitude=46.2, longitude=6.15
            ),
            dropoff_location=Location(
                address="456 Oak Ave", latitude=46.3, longitude=6.16
            ),
            status=BookingStatus("PENDING"),
            amount=Amount(TEST_AMOUNT),
        )

        assert booking.id.value == 1
        assert booking.customer_name == "Test Customer"
        assert booking.status.value == "PENDING"
        assert booking.amount.value == TEST_AMOUNT

    def test_booking_cancel(self):
        """Test annulation d'un booking."""
        booking = Booking(
            id=BookingId(1),
            company_id=1,
            client_id=1,
            user_id=1,
            customer_name="Test Customer",
            pickup_location=Location(
                address="123 Main St", latitude=46.2, longitude=6.15
            ),
            dropoff_location=Location(
                address="456 Oak Ave", latitude=46.3, longitude=6.16
            ),
            status=BookingStatus("PENDING"),
            amount=Amount(TEST_AMOUNT),
        )

        booking.cancel()
        assert booking.status.value == "CANCELLED"

    def test_booking_cancel_invalid_status(self):
        """Test qu'on ne peut pas annuler un booking COMPLETED."""
        booking = Booking(
            id=BookingId(1),
            company_id=1,
            client_id=1,
            user_id=1,
            customer_name="Test Customer",
            pickup_location=Location(
                address="123 Main St", latitude=46.2, longitude=6.15
            ),
            dropoff_location=Location(
                address="456 Oak Ave", latitude=46.3, longitude=6.16
            ),
            status=BookingStatus("COMPLETED"),
            amount=Amount(TEST_AMOUNT),
        )

        with pytest.raises(ValueError, match="Cannot cancel"):
            booking.cancel()

    def test_booking_assign_to_driver(self):
        """Test assignation d'un booking à un driver."""
        booking = Booking(
            id=BookingId(1),
            company_id=1,
            client_id=1,
            user_id=1,
            customer_name="Test Customer",
            pickup_location=Location(
                address="123 Main St", latitude=46.2, longitude=6.15
            ),
            dropoff_location=Location(
                address="456 Oak Ave", latitude=46.3, longitude=6.16
            ),
            status=BookingStatus("PENDING"),
            amount=Amount(TEST_AMOUNT),
        )

        booking.assign_to_driver(driver_id=TEST_DRIVER_ID)
        assert booking.status.value == "ASSIGNED"
        assert booking.driver_id == TEST_DRIVER_ID

    def test_booking_validate(self):
        """Test validation des invariants."""
        booking = Booking(
            id=BookingId(1),
            company_id=1,
            client_id=1,
            user_id=1,
            customer_name="Test Customer",
            pickup_location=Location(
                address="123 Main St", latitude=46.2, longitude=6.15
            ),
            dropoff_location=Location(
                address="456 Oak Ave", latitude=46.3, longitude=6.16
            ),
            status=BookingStatus("PENDING"),
            amount=Amount(TEST_AMOUNT),
        )

        assert booking.validate() is True

    def test_booking_validate_invalid_assigned_no_driver(self):
        """Test validation échoue si ASSIGNED sans driver_id."""
        booking = Booking(
            id=BookingId(1),
            company_id=1,
            client_id=1,
            user_id=1,
            customer_name="Test Customer",
            pickup_location=Location(
                address="123 Main St", latitude=46.2, longitude=6.15
            ),
            dropoff_location=Location(
                address="456 Oak Ave", latitude=46.3, longitude=6.16
            ),
            status=BookingStatus("ASSIGNED"),
            amount=Amount(TEST_AMOUNT),
            driver_id=None,  # Invalid: ASSIGNED requires driver_id
        )

        assert booking.validate() is False
