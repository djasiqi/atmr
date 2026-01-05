# backend/tests/services/unified_dispatch/orchestration/test_result_builder.py
"""Tests unitaires pour ResultBuilder.

Tests pour :
- _serialize_assignment, _serialize_booking, _serialize_driver : Sérialisation
- build : Construction du résultat final avec DispatchResult
"""

from __future__ import annotations  # noqa: I001

import pytest
from unittest.mock import MagicMock, patch

from services.unified_dispatch.orchestration.result_builder import ResultBuilder


class TestSerialize:
    """Tests pour les méthodes de sérialisation."""

    def test_serialize_assignment_with_to_dict(self):
        """Test : Sérialisation assignment avec to_dict()."""
        builder = ResultBuilder()

        assignment = MagicMock()
        assignment.to_dict.return_value = {"booking_id": 1, "driver_id": 2}

        result = builder._serialize_assignment(assignment)

        assert result == {"booking_id": 1, "driver_id": 2}

    def test_serialize_assignment_without_to_dict(self):
        """Test : Sérialisation assignment sans to_dict() (fallback)."""
        builder = ResultBuilder()

        assignment = MagicMock()
        del assignment.to_dict
        assignment.booking_id = 1
        assignment.driver_id = 2
        assignment.dispatch_run_id = 3

        result = builder._serialize_assignment(assignment)

        assert result == {
            "booking_id": 1,
            "driver_id": 2,
            "dispatch_run_id": 3,
        }

    def test_serialize_booking_with_to_dict(self):
        """Test : Sérialisation booking avec to_dict()."""
        builder = ResultBuilder()

        booking = MagicMock()
        booking.to_dict.return_value = {"id": 1, "pickup_lat": 45.0}

        result = builder._serialize_booking(booking)

        assert result == {"id": 1, "pickup_lat": 45.0}

    def test_serialize_booking_without_to_dict(self):
        """Test : Sérialisation booking sans to_dict() (fallback)."""
        builder = ResultBuilder()

        booking = MagicMock()
        del booking.to_dict
        booking.id = 1
        booking.pickup_lat = 45.0
        booking.pickup_lon = -73.0
        booking.dropoff_lat = 46.0
        booking.dropoff_lon = -74.0

        result = builder._serialize_booking(booking)

        assert result == {
            "id": 1,
            "pickup_lat": 45.0,
            "pickup_lon": -73.0,
            "dropoff_lat": 46.0,
            "dropoff_lon": -74.0,
        }

    def test_serialize_driver_with_to_dict(self):
        """Test : Sérialisation driver avec to_dict()."""
        builder = ResultBuilder()

        driver = MagicMock()
        driver.to_dict.return_value = {"id": 1, "current_lat": 45.0}

        result = builder._serialize_driver(driver)

        assert result == {"id": 1, "current_lat": 45.0}

    def test_serialize_driver_without_to_dict(self):
        """Test : Sérialisation driver sans to_dict() (fallback)."""
        builder = ResultBuilder()

        driver = MagicMock()
        del driver.to_dict
        driver.id = 1
        driver.current_lat = 45.0
        driver.current_lon = -73.0

        result = builder._serialize_driver(driver)

        assert result == {
            "id": 1,
            "current_lat": 45.0,
            "current_lon": -73.0,
        }


class TestBuild:
    """Tests pour la méthode build."""

    @patch("services.unified_dispatch.orchestration.result_builder.DispatchResult")
    def test_build_success(self, mock_dispatch_result_class):
        """Test : Construction réussie avec toutes les données."""
        builder = ResultBuilder()

        mock_result_instance = MagicMock()
        mock_result_instance.to_dict.return_value = {
            "dispatch_run_id": 42,
            "assignments": [{"booking_id": 1, "driver_id": 2}],
            "unassigned": [3],
            "bookings": [{"id": 1}],
            "drivers": [{"id": 1}],
            "meta": {"assignments_count": 1},
            "debug": {"used_heuristic": True},
        }
        mock_dispatch_result_class.return_value = mock_result_instance

        assignments = [MagicMock(booking_id=1, driver_id=2)]
        bookings = [MagicMock(id=1)]
        drivers = [MagicMock(id=1)]
        meta = {"assignments_count": 1}
        debug = {"used_heuristic": True}

        result = builder.build(
            dispatch_run_id=42,
            assignments=assignments,
            unassigned_ids=[3],
            bookings=bookings,
            drivers=drivers,
            meta=meta,
            debug=debug,
        )

        assert result["dispatch_run_id"] == 42
        assert len(result["assignments"]) == 1
        assert result["unassigned"] == [3]
        mock_dispatch_result_class.assert_called_once()
        mock_result_instance.to_dict.assert_called_once()

    @patch("services.unified_dispatch.orchestration.result_builder.DispatchResult")
    def test_build_with_dispatch_run_id_none(self, mock_dispatch_result_class):
        """Test : Gestion de dispatch_run_id None."""
        builder = ResultBuilder()

        mock_result_instance = MagicMock()
        mock_result_instance.to_dict.return_value = {
            "dispatch_run_id": None,
            "assignments": [],
            "unassigned": [],
            "bookings": [],
            "drivers": [],
            "meta": {},
            "debug": {},
        }
        mock_dispatch_result_class.return_value = mock_result_instance

        result = builder.build(
            dispatch_run_id=None,
            assignments=[],
            unassigned_ids=[],
            bookings=[],
            drivers=[],
            meta={},
            debug={},
        )

        assert result["dispatch_run_id"] is None

    @patch("services.unified_dispatch.orchestration.result_builder.DispatchResult")
    def test_build_serializes_entities(self, mock_dispatch_result_class):
        """Test : Vérification que les entités sont correctement sérialisées."""
        builder = ResultBuilder()

        mock_result_instance = MagicMock()
        mock_result_instance.to_dict.return_value = {}
        mock_dispatch_result_class.return_value = mock_result_instance

        assignment = MagicMock()
        assignment.booking_id = 1
        assignment.driver_id = 2
        del assignment.to_dict

        booking = MagicMock()
        booking.id = 1
        booking.pickup_lat = 45.0
        del booking.to_dict

        driver = MagicMock()
        driver.id = 1
        driver.current_lat = 45.0
        del driver.to_dict

        builder.build(
            dispatch_run_id=42,
            assignments=[assignment],
            unassigned_ids=[],
            bookings=[booking],
            drivers=[driver],
            meta={},
            debug={},
        )

        # Vérifier que DispatchResult est appelé avec les entités sérialisées
        call_args = mock_dispatch_result_class.call_args
        assert call_args is not None
        assert call_args.kwargs["dispatch_run_id"] == 42
        assert len(call_args.kwargs["assignments"]) == 1
        assert call_args.kwargs["assignments"][0]["booking_id"] == 1
        assert call_args.kwargs["assignments"][0]["driver_id"] == 2
        assert len(call_args.kwargs["bookings"]) == 1
        assert call_args.kwargs["bookings"][0]["id"] == 1
        assert len(call_args.kwargs["drivers"]) == 1
        assert call_args.kwargs["drivers"][0]["id"] == 1
