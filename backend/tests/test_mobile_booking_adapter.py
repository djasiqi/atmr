"""Tests pour l'adaptateur mobile → payload canonique ManualBookingCreateSchema."""

import pytest

from services.adapters.mobile_booking_adapter import (
    map_mobile_ride_payload_to_manual_booking_payload,
)


class TestMapMobileRidePayloadToManualBookingPayload:
    """Tests de map_mobile_ride_payload_to_manual_booking_payload."""

    def test_round_trip_without_return_time_derives_return_date_from_scheduled(self):
        """Si is_return=True et return_time absent, dériver return_date de scheduled_time."""
        payload = {
            "client_id": 1,
            "pickup_address": "Adresse A",
            "dropoff_address": "Adresse B",
            "scheduled_time": "2026-03-15T18:11:00",
            "is_return": True,
        }
        result = map_mobile_ride_payload_to_manual_booking_payload(payload)
        assert result.get("is_round_trip") is True
        assert result.get("return_date") == "2026-03-15"
        assert "return_time" not in result or result.get("return_time") is None

    def test_round_trip_with_return_time_date_only(self):
        """return_time au format YYYY-MM-DD → return_date sans return_time."""
        payload = {
            "client_id": 1,
            "pickup_address": "A",
            "dropoff_address": "B",
            "scheduled_time": "2026-03-15T10:00:00",
            "is_return": True,
            "return_time": "2026-03-15",
        }
        result = map_mobile_ride_payload_to_manual_booking_payload(payload)
        assert result.get("return_date") == "2026-03-15"
        assert "return_time" not in result or result.get("return_time") is None

    def test_round_trip_with_return_time_datetime(self):
        """return_time au format datetime → return_date + return_time."""
        payload = {
            "client_id": 1,
            "pickup_address": "A",
            "dropoff_address": "B",
            "scheduled_time": "2026-03-15T10:00:00",
            "is_return": True,
            "return_time": "2026-03-15T16:30:00",
        }
        result = map_mobile_ride_payload_to_manual_booking_payload(payload)
        assert result.get("return_date") == "2026-03-15"
        assert result.get("return_time") == "2026-03-15T16:30:00"

    def test_round_trip_scheduled_time_date_only_fallback(self):
        """scheduled_time au format date seule (YYYY-MM-DD) → return_date dérivé."""
        payload = {
            "client_id": 1,
            "pickup_address": "A",
            "dropoff_address": "B",
            "scheduled_time": "2026-03-15",
            "is_return": True,
        }
        result = map_mobile_ride_payload_to_manual_booking_payload(payload)
        assert result.get("return_date") == "2026-03-15"

    def test_round_trip_with_return_date_direct(self):
        """Mobile envoie return_date directement (format web) → priorité."""
        payload = {
            "client_id": 1,
            "pickup_address": "A",
            "dropoff_address": "B",
            "scheduled_time": "2026-03-15T10:00:00",
            "is_return": True,
            "return_date": "2026-03-15",
        }
        result = map_mobile_ride_payload_to_manual_booking_payload(payload)
        assert result.get("return_date") == "2026-03-15"
        assert "return_time" not in result or result.get("return_time") is None

    def test_round_trip_with_return_date_and_return_time(self):
        """Mobile envoie return_date + return_time (heure fixée) → les deux."""
        payload = {
            "client_id": 1,
            "pickup_address": "A",
            "dropoff_address": "B",
            "scheduled_time": "2026-03-15T10:00:00",
            "is_return": True,
            "return_date": "2026-03-15",
            "return_time": "2026-03-15T16:30:00",
        }
        result = map_mobile_ride_payload_to_manual_booking_payload(payload)
        assert result.get("return_date") == "2026-03-15"
        assert result.get("return_time") == "2026-03-15T16:30:00"
