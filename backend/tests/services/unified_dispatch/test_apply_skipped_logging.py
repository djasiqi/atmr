# backend/tests/services/unified_dispatch/test_apply_skipped_logging.py
"""Tests pour valider le logging détaillé des assignations skipped."""

from unittest.mock import patch

import pytest
from flask import Flask

from models import BookingStatus
from services.unified_dispatch.optimization.assignment_applier import apply_assignments, logger
from tests.factories import BookingFactory, CompanyFactory, DriverFactory


@pytest.fixture(autouse=True)
def _app_context(app: Flask):
    """Assure que tous les tests s'exécutent dans un app context."""
    with app.app_context():
        yield


@pytest.fixture
def company(db):
    """Créer une entreprise pour les tests."""
    company = CompanyFactory()
    db.session.flush()
    return company


@pytest.fixture
def driver(db, company):
    """Créer un chauffeur pour les tests."""
    driver = DriverFactory(company=company, is_active=True, is_available=True)
    db.session.flush()
    return driver


class TestSkippedAssignmentsLogging:
    """Tests pour valider le logging détaillé des assignations skipped."""

    def test_skip_booking_not_found_logged(self, db, company, driver):
        """Test que skip 'booking_not_found_or_wrong_company' est loggé."""
        # Créer une assignation avec un booking_id qui ne sera pas trouvé
        assignments = [
            type(
                "Assignment",
                (),
                {
                    "booking_id": 99999,  # Booking inexistant
                    "driver_id": driver.id,
                    "score": 1.0,
                },
            )()
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = apply_assignments(
                company_id=company.id,
                assignments=assignments,
            )

            # Vérifier que le booking a été skipped
            assert 99999 in result["skipped"]
            assert result["skipped"][99999] == "booking_not_found_or_wrong_company"

            # Vérifier qu'un log warning a été généré
            assert mock_warning.called
            # Vérifier que le log contient les informations attendues
            call_args_str = str(mock_warning.call_args_list)
            assert "99999" in call_args_str or "booking_id=99999" in call_args_str
            assert "driver_id" in call_args_str.lower()
            assert "booking_not_found_or_wrong_company" in call_args_str

    def test_skip_status_invalid_logged(self, db, company, driver):
        """Test que skip 'status_is_*' est loggé avec métadonnées."""
        # Créer un booking avec un statut invalide (CANCELED)
        booking = BookingFactory(company=company, status=BookingStatus.CANCELED)
        db.session.commit()

        assignments = [
            type(
                "Assignment",
                (),
                {
                    "booking_id": booking.id,
                    "driver_id": driver.id,
                    "score": 1.0,
                },
            )()
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = apply_assignments(
                company_id=company.id,
                assignments=assignments,
            )

            # Vérifier que le booking a été skipped
            assert booking.id in result["skipped"]
            assert "status_is_" in result["skipped"][booking.id]

            # Vérifier qu'un log warning a été généré avec métadonnées
            assert mock_warning.called
            call_args_str = str(mock_warning.call_args_list)
            assert str(booking.id) in call_args_str
            assert "scheduled_time" in call_args_str.lower()
            assert "time_confirmed" in call_args_str.lower()
            assert "is_return" in call_args_str.lower()

    def test_skip_driver_not_found_logged(self, db, company):
        """Test que skip 'driver_not_found_or_wrong_company' est loggé."""
        # Créer un booking valide
        booking = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        db.session.commit()

        # Créer une assignation avec un driver_id inexistant
        assignments = [
            type(
                "Assignment",
                (),
                {
                    "booking_id": booking.id,
                    "driver_id": 99999,  # Driver inexistant
                    "score": 1.0,
                },
            )()
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = apply_assignments(
                company_id=company.id,
                assignments=assignments,
            )

            # Vérifier que le booking a été skipped
            assert booking.id in result["skipped"]
            assert result["skipped"][booking.id] == "driver_not_found_or_wrong_company"

            # Vérifier qu'un log warning a été généré
            assert mock_warning.called
            call_args_str = str(mock_warning.call_args_list)
            assert str(booking.id) in call_args_str
            assert "99999" in call_args_str  # driver_id
            assert "driver_not_found_or_wrong_company" in call_args_str

    def test_skip_driver_not_available_logged(self, db, company, driver):
        """Test que skip 'driver_not_available' est loggé avec état driver."""
        # Créer un driver non disponible
        driver.is_active = False
        driver.is_available = False
        db.session.commit()

        # Créer un booking valide
        booking = BookingFactory(company=company, status=BookingStatus.ACCEPTED)
        db.session.commit()

        assignments = [
            type(
                "Assignment",
                (),
                {
                    "booking_id": booking.id,
                    "driver_id": driver.id,
                    "score": 1.0,
                },
            )()
        ]

        with patch.object(logger, "warning") as mock_warning:
            result = apply_assignments(
                company_id=company.id,
                assignments=assignments,
                enforce_driver_checks=True,  # Activer les vérifications
            )

            # Vérifier que le booking a été skipped
            assert booking.id in result["skipped"]
            assert result["skipped"][booking.id] == "driver_not_available"

            # Vérifier qu'un log warning a été généré avec état driver
            assert mock_warning.called
            call_args_str = str(mock_warning.call_args_list)
            assert str(booking.id) in call_args_str
            assert "driver_not_available" in call_args_str
            assert "driver_is_active" in call_args_str.lower()
            assert "driver_is_available" in call_args_str.lower()
