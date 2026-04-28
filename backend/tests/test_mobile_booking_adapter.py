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

    def test_structured_addresses_are_mapped_to_location_and_coordinates(self):
        payload = {
            "client_id": 1,
            "pickup_address": {
                "label": "Rue de la Gare 1, 1000 Lausanne",
                "place_id": "pickup-place",
                "lat": 46.5197,
                "lon": 6.6323,
            },
            "dropoff_address": {
                "label": "Avenue de la Sallaz 10, 1010 Lausanne",
                "place_id": "dropoff-place",
                "lat": 46.5402,
                "lon": 6.6582,
            },
            "scheduled_time": "2026-03-15T10:00:00",
        }
        result = map_mobile_ride_payload_to_manual_booking_payload(payload)
        assert result["pickup_location"] == "Rue de la Gare 1, 1000 Lausanne"
        assert result["dropoff_location"] == "Avenue de la Sallaz 10, 1010 Lausanne"
        assert result["pickup_lat"] == pytest.approx(46.5197)
        assert result["pickup_lon"] == pytest.approx(6.6323)
        assert result["dropoff_lat"] == pytest.approx(46.5402)
        assert result["dropoff_lon"] == pytest.approx(6.6582)
        assert result["pickup_place_id"] == "pickup-place"
        assert result["dropoff_place_id"] == "dropoff-place"

    def test_structured_address_mode_can_enforce_object_shape(self):
        payload = {
            "client_id": 1,
            "pickup_address": "Adresse legacy",
            "dropoff_address": {"label": "Adresse B"},
            "scheduled_time": "2026-03-15T10:00:00",
        }
        with pytest.raises(ValueError, match=r"pickup_address\.label est requis"):
            map_mobile_ride_payload_to_manual_booking_payload(
                payload, enforce_structured_address=True
            )
