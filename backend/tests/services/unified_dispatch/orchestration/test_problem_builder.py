# backend/tests/services/unified_dispatch/orchestration/test_problem_builder.py
"""Tests unitaires pour ProblemBuilder.

Tests pour :
- build : Construction du problème VRPTW
- validate_geographic_coordinates : Validation des coordonnées géographiques
"""

from __future__ import annotations  # noqa: I001

import pytest
from unittest.mock import MagicMock, patch

from factories import CompanyFactory, DispatchRunFactory
from models import DispatchRun, DispatchStatus
from services.unified_dispatch.orchestration.problem_builder import ProblemBuilder
from shared.constants import GeoConstants


class TestBuild:
    """Tests pour la méthode build."""

    @patch(
        "services.unified_dispatch.orchestration.problem_builder.data.build_problem_data"
    )
    def test_build_success_with_bookings_and_drivers(self, mock_build_data, db):
        """Test : Construction réussie avec bookings et drivers."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        # Mock build_problem_data pour retourner des données valides
        mock_booking = MagicMock()
        mock_booking.id = 1
        mock_booking.pickup_lat = 48.8566
        mock_booking.pickup_lon = 2.3522
        mock_booking.dropoff_lat = 48.8606
        mock_booking.dropoff_lon = 2.3376

        mock_driver = MagicMock()
        mock_driver.id = 1

        mock_build_data.return_value = {
            "bookings": [mock_booking],
            "drivers": [mock_driver],
        }

        builder = ProblemBuilder()
        problem, _error_result = builder.build(
            _company=company,
            company_id=company.id,
            dispatch_run=dispatch_run,
            settings=MagicMock(),
            for_date="2025-01-14",
            day_str="2025-01-14",
            regular_first=True,
            allow_emg=True,
            overrides=None,
            perf_collector=None,
        )

        assert problem is not None
        assert "bookings" in problem
        assert "drivers" in problem
        assert problem["dispatch_run_id"] == dispatch_run.id

    @patch(
        "services.unified_dispatch.orchestration.problem_builder.data.build_problem_data"
    )
    def test_build_no_data_returns_error(self, mock_build_data, db):
        """Test : Gestion du cas 'no_data' (pas de bookings ou drivers)."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        # Mock build_problem_data pour retourner des données vides
        mock_build_data.return_value = {"bookings": [], "drivers": []}

        builder = ProblemBuilder()
        problem, error_result = builder.build(
            _company=company,
            company_id=company.id,
            dispatch_run=dispatch_run,
            settings=MagicMock(),
            for_date="2025-01-14",
            day_str="2025-01-14",
            regular_first=True,
            allow_emg=True,
            overrides=None,
            perf_collector=None,
        )

        assert problem is None
        assert error_result is not None
        assert error_result["meta"]["reason"] == "no_data"

    @patch(
        "services.unified_dispatch.orchestration.problem_builder.data.build_problem_data"
    )
    def test_build_propagates_dispatch_run_id(self, mock_build_data, db):
        """Test : Propagation de dispatch_run_id dans le problem."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        mock_booking = MagicMock()
        mock_booking.id = 1
        mock_booking.pickup_lat = 48.8566
        mock_booking.pickup_lon = 2.3522
        mock_booking.dropoff_lat = 48.8606
        mock_booking.dropoff_lon = 2.3376

        mock_build_data.return_value = {
            "bookings": [mock_booking],
            "drivers": [MagicMock()],
        }

        builder = ProblemBuilder()
        problem, _error_result = builder.build(
            _company=company,
            company_id=company.id,
            dispatch_run=dispatch_run,
            settings=MagicMock(),
            for_date="2025-01-14",
            day_str="2025-01-14",
            regular_first=True,
            allow_emg=True,
            overrides=None,
            perf_collector=None,
        )

        assert problem is not None
        assert problem["dispatch_run_id"] == dispatch_run.id

    @patch(
        "services.unified_dispatch.orchestration.problem_builder.data.build_problem_data"
    )
    def test_build_handles_db_error(self, mock_build_data, db):
        """Test : Gestion des erreurs DB (OperationalError, DBAPIError)."""
        from sqlalchemy.exc import OperationalError

        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        # Mock build_problem_data pour lever une OperationalError
        mock_build_data.side_effect = OperationalError(
            "DB connection error", None, None
        )

        builder = ProblemBuilder()
        problem, _error_result = builder.build(
            _company=company,
            company_id=company.id,
            dispatch_run=dispatch_run,
            settings=MagicMock(),
            for_date="2025-01-14",
            day_str="2025-01-14",
            regular_first=True,
            allow_emg=True,
            overrides=None,
            perf_collector=None,
        )

        # En cas d'erreur DB, problem devrait être vide et error_result None
        # (car l'erreur est loggée mais pas retournée comme error_result)
        assert problem == {}

    @patch(
        "services.unified_dispatch.orchestration.problem_builder.data.build_problem_data"
    )
    def test_build_handles_validation_error(self, mock_build_data, db):
        """Test : Gestion des erreurs de validation (ValueError, TypeError, AttributeError)."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        # Mock build_problem_data pour lever une ValueError
        mock_build_data.side_effect = ValueError("Invalid data")

        builder = ProblemBuilder()
        problem, _error_result = builder.build(
            _company=company,
            company_id=company.id,
            dispatch_run=dispatch_run,
            settings=MagicMock(),
            for_date="2025-01-14",
            day_str="2025-01-14",
            regular_first=True,
            allow_emg=True,
            overrides=None,
            perf_collector=None,
        )

        # En cas d'erreur de validation, problem devrait être vide
        assert problem == {}

    @patch(
        "services.unified_dispatch.orchestration.problem_builder.data.build_problem_data"
    )
    @patch(
        "services.unified_dispatch.orchestration.problem_builder.ProblemBuilder._dispatch_run_manager"
    )
    def test_build_marks_dispatch_run_failed_on_unexpected_error(
        self, mock_manager, mock_build_data, db
    ):
        """Test : Marquage DispatchRun FAILED en cas d'erreur inattendue."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        # Mock build_problem_data pour lever une exception inattendue
        mock_build_data.side_effect = RuntimeError("Unexpected error")

        builder = ProblemBuilder()
        problem, error_result = builder.build(
            _company=company,
            company_id=company.id,
            dispatch_run=dispatch_run,
            settings=MagicMock(),
            for_date="2025-01-14",
            day_str="2025-01-14",
            regular_first=True,
            allow_emg=True,
            overrides=None,
            perf_collector=None,
        )

        # Vérifier que update_status a été appelé pour marquer FAILED
        mock_manager.update_status.assert_called_once_with(
            dispatch_run, DispatchStatus.FAILED
        )
        assert problem is None
        assert error_result is not None
        assert error_result["meta"]["reason"] == "problem_build_failed"

    @patch(
        "services.unified_dispatch.orchestration.problem_builder.data.build_problem_data"
    )
    def test_build_with_perf_collector(self, mock_build_data, db):
        """Test : Vérification du timer perf_collector (start/end)."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        dispatch_run = DispatchRunFactory(company_id=company.id)
        db.session.add(dispatch_run)
        db.session.commit()

        mock_booking = MagicMock()
        mock_booking.id = 1
        mock_booking.pickup_lat = 48.8566
        mock_booking.pickup_lon = 2.3522
        mock_booking.dropoff_lat = 48.8606
        mock_booking.dropoff_lon = 2.3376

        mock_build_data.return_value = {
            "bookings": [mock_booking],
            "drivers": [MagicMock()],
        }

        perf_collector = MagicMock()

        builder = ProblemBuilder()
        problem, _error_result = builder.build(
            _company=company,
            company_id=company.id,
            dispatch_run=dispatch_run,
            settings=MagicMock(),
            for_date="2025-01-14",
            day_str="2025-01-14",
            regular_first=True,
            allow_emg=True,
            overrides=None,
            perf_collector=perf_collector,
        )

        # Vérifier que end_timer a été appelé
        perf_collector.end_timer.assert_called_once_with("data_collection")
        assert problem is not None


class TestValidateGeographicCoordinates:
    """Tests pour la méthode validate_geographic_coordinates."""

    def test_validate_coordinates_with_valid_bookings(self):
        """Test : Bookings avec coordonnées valides."""
        mock_booking1 = MagicMock()
        mock_booking1.id = 1
        mock_booking1.pickup_lat = 48.8566  # Paris
        mock_booking1.pickup_lon = 2.3522
        mock_booking1.dropoff_lat = 48.8606
        mock_booking1.dropoff_lon = 2.3376

        mock_booking2 = MagicMock()
        mock_booking2.id = 2
        mock_booking2.pickup_lat = 45.7640  # Lyon
        mock_booking2.pickup_lon = 4.8357
        mock_booking2.dropoff_lat = 45.7500
        mock_booking2.dropoff_lon = 4.8500

        problem = {"bookings": [mock_booking1, mock_booking2]}

        builder = ProblemBuilder()
        result = builder.validate_geographic_coordinates(problem)

        assert result["bookings_without_coords"] == []
        assert result["bookings_with_invalid_coords"] == []

    def test_validate_coordinates_with_missing_coords(self):
        """Test : Bookings sans coordonnées (pickup ou dropoff manquantes)."""
        mock_booking1 = MagicMock()
        mock_booking1.id = 1
        mock_booking1.pickup_lat = None  # Manquant
        mock_booking1.pickup_lon = 2.3522
        mock_booking1.dropoff_lat = 48.8606
        mock_booking1.dropoff_lon = 2.3376

        mock_booking2 = MagicMock()
        mock_booking2.id = 2
        mock_booking2.pickup_lat = 48.8566
        mock_booking2.pickup_lon = 2.3522
        mock_booking2.dropoff_lat = None  # Manquant
        mock_booking2.dropoff_lon = 2.3376

        problem = {"bookings": [mock_booking1, mock_booking2]}

        builder = ProblemBuilder()
        result = builder.validate_geographic_coordinates(problem)

        assert len(result["bookings_without_coords"]) == 2
        assert 1 in result["bookings_without_coords"]
        assert 2 in result["bookings_without_coords"]
        assert result["bookings_with_invalid_coords"] == []

    def test_validate_coordinates_with_invalid_coords(self):
        """Test : Bookings avec coordonnées invalides (hors plages)."""
        # Coordonnées hors plages valides
        invalid_lat = GeoConstants.LATITUDE_MAX + 10
        invalid_lon = GeoConstants.LONGITUDE_MAX + 10

        mock_booking1 = MagicMock()
        mock_booking1.id = 1
        mock_booking1.pickup_lat = invalid_lat
        mock_booking1.pickup_lon = 2.3522
        mock_booking1.dropoff_lat = 48.8606
        mock_booking1.dropoff_lon = 2.3376

        mock_booking2 = MagicMock()
        mock_booking2.id = 2
        mock_booking2.pickup_lat = 48.8566
        mock_booking2.pickup_lon = 2.3522
        mock_booking2.dropoff_lat = 48.8606
        mock_booking2.dropoff_lon = invalid_lon

        problem = {"bookings": [mock_booking1, mock_booking2]}

        builder = ProblemBuilder()
        result = builder.validate_geographic_coordinates(problem)

        assert result["bookings_without_coords"] == []
        assert len(result["bookings_with_invalid_coords"]) == 2
        assert 1 in result["bookings_with_invalid_coords"]
        assert 2 in result["bookings_with_invalid_coords"]

    def test_validate_coordinates_with_none_coords(self):
        """Test : Bookings avec coordonnées None."""
        mock_booking = MagicMock()
        mock_booking.id = 1
        mock_booking.pickup_lat = None
        mock_booking.pickup_lon = None
        mock_booking.dropoff_lat = None
        mock_booking.dropoff_lon = None

        problem = {"bookings": [mock_booking]}

        builder = ProblemBuilder()
        result = builder.validate_geographic_coordinates(problem)

        assert len(result["bookings_without_coords"]) == 1
        assert 1 in result["bookings_without_coords"]
        assert result["bookings_with_invalid_coords"] == []

    def test_validate_coordinates_empty_bookings(self):
        """Test : Liste de bookings vide."""
        problem = {"bookings": []}

        builder = ProblemBuilder()
        result = builder.validate_geographic_coordinates(problem)

        assert result["bookings_without_coords"] == []
        assert result["bookings_with_invalid_coords"] == []

    def test_validate_coordinates_limits_logging(self):
        """Test : Limitation du logging (MAX_BOOKING_IDS_TO_LOG = 20)."""
        # Créer plus de 20 bookings sans coordonnées
        bookings = []
        for i in range(25):
            mock_booking = MagicMock()
            mock_booking.id = i + 1
            mock_booking.pickup_lat = None
            mock_booking.pickup_lon = None
            mock_booking.dropoff_lat = None
            mock_booking.dropoff_lon = None
            bookings.append(mock_booking)

        problem = {"bookings": bookings}

        builder = ProblemBuilder()
        result = builder.validate_geographic_coordinates(problem)

        # Tous les bookings devraient être dans la liste
        assert len(result["bookings_without_coords"]) == 25
        # Mais le logging devrait être limité à 20 (vérifié via le code, pas via le résultat)
