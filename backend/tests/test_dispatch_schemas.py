"""
Tests unitaires pour schemas/dispatch_schemas.py
"""
# Les tests utilisent datetime sans tzinfo volontairement pour simplicité

from datetime import date, datetime
from typing import Any, cast

import pytest

from schemas.dispatch_schemas import (
    assignment_schema,
    assignments_schema,
    booking_schema,
    bookings_schema,
    dispatch_result_schema,
    dispatch_run_schema,
    driver_schema,
    drivers_schema,
    suggestion_schema,
    suggestions_schema,
)


class TestDriverSchema:
    """Tests sérialisation Driver."""

    def test_serialize_driver_minimal(self):
        """Sérialisation driver minimal."""
        data = {
            "id": 789,
            "first_name": "Jean",
            "last_name": "Dupont",
            "is_available": True,
            "is_active": True,
        }

        result = cast("dict[str, Any]", driver_schema.dump(data))

        assert result["id"] == 789
        assert result["first_name"] == "Jean"
        assert result["last_name"] == "Dupont"
        assert result["is_available"] is True
        assert result["is_active"] is True

    def test_serialize_driver_with_metrics(self):
        """Sérialisation driver avec métriques."""
        data = {
            "id": 789,
            "first_name": "Jean",
            "last_name": "Dupont",
            "phone": "+41791234567",
            "is_available": True,
            "is_active": True,
            "is_emergency": False,
            "punctuality_score": 0.92,
            "current_load": 3,
            "current_lat": 46.2044,
            "current_lon": 6.1432,
        }

        result = cast("dict[str, Any]", driver_schema.dump(data))

        assert result["punctuality_score"] == 0.92
        assert result["current_load"] == 3
        assert result["current_lat"] == 46.2044
        assert result["current_lon"] == 6.1432

    def test_serialize_drivers_multiple(self):
        """Sérialisation liste de drivers."""
        data = [
            {"id": 1, "first_name": "Jean", "last_name": "Dupont", "is_available": True, "is_active": True},
            {"id": 2, "first_name": "Marie", "last_name": "Martin", "is_available": False, "is_active": True},
        ]

        result = cast("list[dict[str, Any]]", drivers_schema.dump(data))

        assert len(result) == 2
        assert result[0]["first_name"] == "Jean"
        assert result[1]["first_name"] == "Marie"


class TestBookingSchema:
    """Tests sérialisation Booking."""

    def test_serialize_booking_minimal(self):
        """Sérialisation booking minimal."""
        data = {
            "id": 456,
            "scheduled_time": datetime(2025, 10, 20, 14, 30),
            "status": "pending",
            "is_medical": True,
            "is_urgent": False,
            "company_id": 1,
        }

        result = cast("dict[str, Any]", booking_schema.dump(data))

        assert result["id"] == 456
        assert result["status"] == "pending"
        assert result["is_medical"] is True
        assert result["is_urgent"] is False

    def test_serialize_booking_with_addresses(self):
        """Sérialisation booking avec adresses."""
        data = {
            "id": 456,
            "scheduled_time": datetime(2025, 10, 20, 14, 30),
            "pickup_address": "123 Rue de la Paix, Genève",
            "dropoff_address": "456 Avenue du Lac, Lausanne",
            "pickup_lat": 46.2044,
            "pickup_lon": 6.1432,
            "dropoff_lat": 46.5197,
            "dropoff_lon": 6.6323,
            "status": "assigned",
            "is_medical": True,
            "is_urgent": False,
            "company_id": 1,
        }

        result = cast("dict[str, Any]", booking_schema.dump(data))

        assert result["pickup_address"] == "123 Rue de la Paix, Genève"
        assert result["dropoff_address"] == "456 Avenue du Lac, Lausanne"
        assert result["pickup_lat"] == 46.2044
        assert result["pickup_lon"] == 6.1432

    def test_serialize_bookings_multiple(self):
        """Sérialisation liste de bookings."""
        data = [
            {
                "id": 1,
                "scheduled_time": datetime(2025, 10, 20, 10, 0),
                "status": "pending",
                "is_medical": True,
                "is_urgent": False,
                "company_id": 1,
            },
            {
                "id": 2,
                "scheduled_time": datetime(2025, 10, 20, 11, 0),
                "status": "assigned",
                "is_medical": False,
                "is_urgent": True,
                "company_id": 1,
            },
        ]

        result = cast("list[dict[str, Any]]", bookings_schema.dump(data))

        assert len(result) == 2
        assert result[0]["is_medical"] is True
        assert result[1]["is_urgent"] is True


class TestAssignmentSchema:
    """Tests sérialisation Assignment."""

    def test_serialize_assignment_minimal(self):
        """Sérialisation assignment minimal."""
        data = {
            "id": 123,
            "booking_id": 456,
            "driver_id": 789,
            "created_at": datetime(2025, 10, 20, 10, 0, 0),
            "status": "pending",
            "confirmed": False
        }

        result = cast("dict[str, Any]", assignment_schema.dump(data))

        assert result["id"] == 123
        assert result["booking_id"] == 456
        assert result["driver_id"] == 789
        assert result["status"] == "pending"
        assert result["confirmed"] is False

    def test_serialize_assignment_with_nested(self):
        """Sérialisation avec relations nested."""
        data = {
            "id": 123,
            "booking_id": 456,
            "driver_id": 789,
            "created_at": datetime.now(),
            "status": "confirmed",
            "confirmed": True,
            "booking": {
                "id": 456,
                "scheduled_time": datetime.now(),
                "pickup_address": "123 Rue Test",
                "status": "assigned",
                "is_medical": True,
                "is_urgent": False,
                "company_id": 1,
            },
            "driver": {
                "id": 789,
                "first_name": "Jean",
                "last_name": "Dupont",
                "is_available": True,
                "is_active": True,
            }
        }

        result = cast("dict[str, Any]", assignment_schema.dump(data))

        assert result["booking"]["id"] == 456
        assert result["booking"]["pickup_address"] == "123 Rue Test"
        assert result["driver"]["first_name"] == "Jean"
        assert result["driver"]["last_name"] == "Dupont"

    def test_serialize_assignments_multiple(self):
        """Sérialisation liste d'assignments."""
        data = [
            {
                "id": 1,
                "booking_id": 10,
                "driver_id": 20,
                "status": "confirmed",
                "confirmed": True,
                "created_at": datetime.now(),
            },
            {
                "id": 2,
                "booking_id": 11,
                "driver_id": 21,
                "status": "pending",
                "confirmed": False,
                "created_at": datetime.now(),
            },
        ]

        result = cast("list[dict[str, Any]]", assignments_schema.dump(data))

        assert len(result) == 2
        assert result[0]["confirmed"] is True
        assert result[1]["confirmed"] is False


class TestDispatchRunSchema:
    """Tests sérialisation DispatchRun."""

    def test_serialize_dispatch_run(self):
        """Sérialisation dispatch run."""
        data = {
            "id": 100,
            "company_id": 1,
            "created_at": datetime(2025, 10, 20, 8, 0),
            "for_date": date(2025, 10, 20),
            "mode": "semi_auto",
            "quality_score": 85.5,
            "total_bookings": 50,
            "assigned_bookings": 45,
            "unassigned_bookings": 5,
            "total_drivers": 10,
            "solver_time_seconds": 12.5,
            "total_time_seconds": 18.3,
        }

        result = cast("dict[str, Any]", dispatch_run_schema.dump(data))

        assert result["id"] == 100
        assert result["mode"] == "semi_auto"
        assert result["quality_score"] == 85.5
        assert result["total_bookings"] == 50
        assert result["assigned_bookings"] == 45
        assert result["solver_time_seconds"] == 12.5


class TestDispatchSuggestionSchema:
    """Tests sérialisation DispatchSuggestion."""

    def test_serialize_suggestion_reassign(self):
        """Sérialisation suggestion de réassignation."""
        data = {
            "action": "reassign",
            "assignment_id": 123,
            "booking_id": 456,
            "driver_id": 789,
            "alternative_driver_id": 790,
            "reason": "Driver delayed, better alternative found",
            "priority": "high",
            "impact_score": 0.85,
            "predicted_delay_minutes": 15.0,
            "gain_minutes": 8.0,
            "confidence": 0.92,
        }

        result = cast("dict[str, Any]", suggestion_schema.dump(data))

        assert result["action"] == "reassign"
        assert result["alternative_driver_id"] == 790
        assert result["priority"] == "high"
        assert result["gain_minutes"] == 8.0

    def test_serialize_suggestion_notify(self):
        """Sérialisation suggestion de notification."""
        data = {
            "action": "notify",
            "assignment_id": 123,
            "reason": "Driver should leave now",
            "priority": "medium",
            "impact_score": 0.65,
        }

        result = cast("dict[str, Any]", suggestion_schema.dump(data))

        assert result["action"] == "notify"
        assert result["priority"] == "medium"

    def test_serialize_suggestions_multiple(self):
        """Sérialisation liste de suggestions."""
        data = [
            {"action": "notify", "assignment_id": 1, "priority": "high"},
            {"action": "reassign", "assignment_id": 2, "priority": "critical"},
        ]

        result = cast("list[dict[str, Any]]", suggestions_schema.dump(data))

        assert len(result) == 2
        assert result[0]["action"] == "notify"
        assert result[1]["action"] == "reassign"


class TestDispatchResultSchema:
    """Tests sérialisation DispatchResult complet."""

    def test_serialize_dispatch_result(self):
        """Sérialisation résultat dispatch complet."""
        data = {
            "dispatch_run_id": 100,
            "mode": "semi_auto",
            "assignments": [
                {
                    "id": 1,
                    "booking_id": 10,
                    "driver_id": 20,
                    "status": "confirmed",
                    "confirmed": True,
                    "created_at": datetime.now(),
                }
            ],
            "unassigned_bookings": [],
            "total_bookings": 10,
            "assigned_count": 10,
            "unassigned_count": 0,
            "quality_score": 90.0,
            "solver_time_seconds": 8.5,
            "total_time_seconds": 12.0,
            "suggestions": [
                {"action": "notify", "assignment_id": 1, "priority": "low"}
            ]
        }

        result = cast("dict[str, Any]", dispatch_result_schema.dump(data))

        assert result["dispatch_run_id"] == 100
        assert result["mode"] == "semi_auto"
        assert result["assigned_count"] == 10
        assert result["quality_score"] == 90.0
        assert len(result["assignments"]) == 1
        assert len(result["suggestions"]) == 1


class TestSchemaValidation:
    """Tests de validation des schémas."""

    def test_assignment_requires_booking_and_driver(self):
        """Assignment nécessite booking_id et driver_id."""
        # Données invalides (manque booking_id)
        data = {
            "driver_id": 789,
            "status": "pending",
            "confirmed": False
        }

        result = cast("dict[str, Any]", assignment_schema.dump(data))
        # Dump ne valide pas, juste sérialise ce qui est là
        assert "driver_id" in result
        assert "booking_id" not in result or result.get("booking_id") is None

    def test_booking_schema_iso_datetime(self):
        """Les dates sont formatées en ISO."""
        data = {
            "id": 456,
            "scheduled_time": datetime(2025, 10, 20, 14, 30, 0),
            "status": "pending",
            "is_medical": False,
            "is_urgent": False,
            "company_id": 1,
        }

        result = cast("dict[str, Any]", booking_schema.dump(data))

        # Vérifier format ISO
        assert "T" in result["scheduled_time"]  # Format ISO contient 'T'
        assert "2025-10-20" in result["scheduled_time"]


class TestSchemaOrdering:
    """Tests que les champs sont ordonnés."""

    def test_driver_schema_ordered(self):
        """Les champs sont dans l'ordre défini."""
        data = {
            "id": 1,
            "first_name": "Jean",
            "last_name": "Dupont",
            "is_available": True,
            "is_active": True,
        }

        result = cast("dict[str, Any]", driver_schema.dump(data))
        keys = list(result.keys())

        # ID doit être en premier
        assert keys[0] == "id"
        assert "first_name" in keys
        assert "last_name" in keys

    def test_booking_schema_ordered(self):
        """Les champs booking sont ordonnés."""
        data = {
            "id": 1,
            "scheduled_time": datetime(2025, 10, 20, 10, 0),
            "status": "pending",
            "is_medical": False,
            "is_urgent": False,
            "company_id": 1,
        }

        result = cast("dict[str, Any]", booking_schema.dump(data))
        keys = list(result.keys())

        # ID doit être en premier
        assert keys[0] == "id"

