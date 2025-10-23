"""
Tests d'intégration pour le dispatch end-to-end
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

from models import Assignment, Booking, Company, Driver
from services.unified_dispatch.engine import run as dispatch_run
from services.unified_dispatch.settings import Settings


class TestDispatchIntegration:
    """Tests d'intégration pour le système de dispatch complet"""

    @pytest.fixture
    def mock_company(self):
        """Fixture pour une entreprise de test"""
        company = Mock(spec=Company)
        company.id = 1
        company.name = "Test Company"
        return company

    @pytest.fixture
    def mock_drivers(self):
        """Fixture pour des chauffeurs de test"""
        drivers = []
        for i in range(3):
            driver = Mock(spec=Driver)
            driver.id = i + 1
            driver.available = True
            driver.capacity = 4
            driver.can_handle_emergency = True
            driver.work_windows = [
                Mock(
                    start=datetime(2024, 1, 15, 8, 0, tzinfo=UTC),
                    end=datetime(2024, 1, 15, 18, 0, tzinfo=UTC)
                )
            ]
            drivers.append(driver)
        return drivers

    @pytest.fixture
    def mock_bookings(self):
        """Fixture pour des réservations de test"""
        bookings = []
        base_time = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)

        for i in range(5):
            booking = Mock(spec=Booking)
            booking.id = i + 1
            booking.scheduled_time = base_time + timedelta(minutes=i * 30)
            booking.pickup_lat = 46.5197 + (i * 0.001)  # Légèrement décalé
            booking.pickup_lon = 6.6323 + (i * 0.001)
            booking.dropoff_lat = 46.5200 + (i * 0.001)
            booking.dropoff_lon = 6.6330 + (i * 0.001)
            booking.capacity_required = 1
            booking.is_emergency = False
            booking.status = "confirmed"
            bookings.append(booking)

        return bookings

    @pytest.fixture
    def default_settings(self):
        """Fixture pour les paramètres par défaut"""
        return Settings()

    @patch('services.unified_dispatch.engine._acquire_day_lock')
    @patch('services.unified_dispatch.engine._release_day_lock')
    @patch('services.unified_dispatch.engine._build_problem')
    @patch('services.unified_dispatch.engine._apply_and_emit')
    def test_dispatch_run_success(
        self,
        mock_apply_and_emit,
        mock_build_problem,
        mock_release_lock,
        mock_acquire_lock,
        mock_company,
        mock_drivers,
        mock_bookings,
        default_settings
    ):
        """Test qu'un run de dispatch se termine avec succès"""
        # Arrange
        mock_acquire_lock.return_value = True
        mock_build_problem.return_value = {
            "bookings": mock_bookings,
            "drivers": mock_drivers
        }

        # Mock des résultats d'heuristique et de solver
        mock_heuristic_result = Mock()
        mock_heuristic_result.assignments = [
            Mock(booking_id=1, driver_id=1),
            Mock(booking_id=2, driver_id=2),
            Mock(booking_id=3, driver_id=1)
        ]
        mock_heuristic_result.unassigned_booking_ids = [4, 5]
        mock_heuristic_result.osrm_calls = 10
        mock_heuristic_result.osrm_avg_latency_ms = 150
        mock_heuristic_result.heuristic_time_ms = 800

        mock_solver_result = Mock()
        mock_solver_result.assignments = []
        mock_solver_result.solver_time_ms = 2000

        with patch('services.unified_dispatch.engine.assign_heuristic') as mock_heuristic, \
             patch('services.unified_dispatch.engine.assign_solver') as mock_solver:

            mock_heuristic.return_value = mock_heuristic_result
            mock_solver.return_value = mock_solver_result

            # Act
            result = dispatch_run(
                company_id=mock_company.id,
                mode="auto",
                regular_first=True,
                allow_emergency=True,
                settings=default_settings
            )

            # Assert
            assert result is not None
            assert "assignments_count" in result
            assert "unassigned_count" in result
            assert "unassigned_reasons" in result
            assert result["assignments_count"] == 3
            assert result["unassigned_count"] == 2

            # Vérifier que les verrous ont été gérés
            mock_acquire_lock.assert_called_once()
            mock_release_lock.assert_called_once()

            # Vérifier que l'heuristique et le solver ont été appelés
            mock_heuristic.assert_called_once()
            mock_solver.assert_called_once()

    @patch('services.unified_dispatch.engine._acquire_day_lock')
    def test_dispatch_run_lock_failure(self, mock_acquire_lock, mock_company, default_settings):
        """Test qu'un run de dispatch échoue si le verrou ne peut pas être acquis"""
        # Arrange
        mock_acquire_lock.return_value = False

        # Act & Assert
        with pytest.raises(Exception, match="Another dispatch is already running"):
            dispatch_run(
                company_id=mock_company.id,
                mode="auto",
                regular_first=True,
                allow_emergency=True,
                settings=default_settings
            )

    @patch('services.unified_dispatch.engine._acquire_day_lock')
    @patch('services.unified_dispatch.engine._release_day_lock')
    @patch('services.unified_dispatch.engine._build_problem')
    def test_dispatch_run_heuristic_only_mode(
        self,
        mock_build_problem,
        mock_release_lock,
        mock_acquire_lock,
        mock_company,
        mock_drivers,
        mock_bookings,
        default_settings
    ):
        """Test qu'un run en mode heuristic_only n'appelle pas le solver"""
        # Arrange
        mock_acquire_lock.return_value = True
        mock_build_problem.return_value = {
            "bookings": mock_bookings,
            "drivers": mock_drivers
        }

        mock_heuristic_result = Mock()
        mock_heuristic_result.assignments = [
            Mock(booking_id=1, driver_id=1),
            Mock(booking_id=2, driver_id=2)
        ]
        mock_heuristic_result.unassigned_booking_ids = [3, 4, 5]
        mock_heuristic_result.osrm_calls = 5
        mock_heuristic_result.osrm_avg_latency_ms = 120
        mock_heuristic_result.heuristic_time_ms = 600

        with patch('services.unified_dispatch.engine.assign_heuristic') as mock_heuristic, \
             patch('services.unified_dispatch.engine.assign_solver') as mock_solver:

            mock_heuristic.return_value = mock_heuristic_result

            # Act
            result = dispatch_run(
                company_id=mock_company.id,
                mode="heuristic_only",
                regular_first=True,
                allow_emergency=True,
                settings=default_settings
            )

            # Assert
            assert result is not None
            assert result["assignments_count"] == 2
            assert result["unassigned_count"] == 3

            # Vérifier que seul l'heuristique a été appelé
            mock_heuristic.assert_called_once()
            mock_solver.assert_not_called()

    @patch('services.unified_dispatch.engine._acquire_day_lock')
    @patch('services.unified_dispatch.engine._release_day_lock')
    @patch('services.unified_dispatch.engine._build_problem')
    def test_dispatch_run_emergency_handling(
        self,
        mock_build_problem,
        mock_release_lock,
        mock_acquire_lock,
        mock_company,
        mock_drivers,
        default_settings
    ):
        """Test que les courses d'urgence sont gérées correctement"""
        # Arrange
        mock_acquire_lock.return_value = True

        # Créer des réservations avec des urgences
        emergency_bookings = []
        base_time = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)

        for i in range(3):
            booking = Mock(spec=Booking)
            booking.id = i + 1
            booking.scheduled_time = base_time + timedelta(minutes=i * 30)
            booking.pickup_lat = 46.5197
            booking.pickup_lon = 6.6323
            booking.dropoff_lat = 46.5200
            booking.dropoff_lon = 6.6330
            booking.capacity_required = 1
            booking.is_emergency = (i == 1)  # Seule la deuxième est une urgence
            booking.status = "confirmed"
            emergency_bookings.append(booking)

        mock_build_problem.return_value = {
            "bookings": emergency_bookings,
            "drivers": mock_drivers
        }

        mock_heuristic_result = Mock()
        mock_heuristic_result.assignments = [
            Mock(booking_id=2, driver_id=1)  # L'urgence est assignée
        ]
        mock_heuristic_result.unassigned_booking_ids = [1, 3]
        mock_heuristic_result.osrm_calls = 3
        mock_heuristic_result.osrm_avg_latency_ms = 100
        mock_heuristic_result.heuristic_time_ms = 400

        with patch('services.unified_dispatch.engine.assign_heuristic') as mock_heuristic, \
             patch('services.unified_dispatch.engine.assign_solver') as mock_solver:

            mock_heuristic.return_value = mock_heuristic_result
            mock_solver.return_value = Mock(assignments=[], solver_time_ms=1000)

            # Act
            result = dispatch_run(
                company_id=mock_company.id,
                mode="auto",
                regular_first=True,
                allow_emergency=True,
                settings=default_settings
            )

            # Assert
            assert result is not None
            assert result["assignments_count"] == 1
            assert result["unassigned_count"] == 2

            # Vérifier que l'heuristique a été appelée avec les bons paramètres
            mock_heuristic.assert_called_once()
            call_args = mock_heuristic.call_args
            assert call_args[1]["allow_emergency"] is True

    def test_dispatch_performance_requirements(self, mock_company, mock_drivers, mock_bookings, default_settings):
        """Test que le dispatch respecte les exigences de performance"""
        # Arrange
        large_bookings = []
        base_time = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)

        # Créer 50 réservations pour tester la performance
        for i in range(50):
            booking = Mock(spec=Booking)
            booking.id = i + 1
            booking.scheduled_time = base_time + timedelta(minutes=i * 10)
            booking.pickup_lat = 46.5197 + (i * 0.0001)
            booking.pickup_lon = 6.6323 + (i * 0.0001)
            booking.dropoff_lat = 46.5200 + (i * 0.0001)
            booking.dropoff_lon = 6.6330 + (i * 0.0001)
            booking.capacity_required = 1
            booking.is_emergency = False
            booking.status = "confirmed"
            large_bookings.append(booking)

        # Créer 10 chauffeurs
        large_drivers = []
        for i in range(10):
            driver = Mock(spec=Driver)
            driver.id = i + 1
            driver.available = True
            driver.capacity = 4
            driver.can_handle_emergency = True
            driver.work_windows = [
                Mock(
                    start=datetime(2024, 1, 15, 8, 0, tzinfo=UTC),
                    end=datetime(2024, 1, 15, 18, 0, tzinfo=UTC)
                )
            ]
            large_drivers.append(driver)

        with patch('services.unified_dispatch.engine._acquire_day_lock') as mock_acquire_lock, \
             patch('services.unified_dispatch.engine._release_day_lock') as _mock_release_lock, \
             patch('services.unified_dispatch.engine._build_problem') as mock_build_problem, \
             patch('services.unified_dispatch.engine._apply_and_emit') as _mock_apply_and_emit:

            mock_acquire_lock.return_value = True
            mock_build_problem.return_value = {
                "bookings": large_bookings,
                "drivers": large_drivers
            }

            mock_heuristic_result = Mock()
            mock_heuristic_result.assignments = [
                Mock(booking_id=i, driver_id=(i % 10) + 1) for i in range(1, 41)
            ]  # 40 assignations
            mock_heuristic_result.unassigned_booking_ids = list(range(41, 51))  # 10 non assignées
            mock_heuristic_result.osrm_calls = 200
            mock_heuristic_result.osrm_avg_latency_ms = 120
            mock_heuristic_result.heuristic_time_ms = 3000

            mock_solver_result = Mock()
            mock_solver_result.assignments = []
            mock_solver_result.solver_time_ms = 1000

            with patch('services.unified_dispatch.engine.assign_heuristic') as mock_heuristic, \
                 patch('services.unified_dispatch.engine.assign_solver') as mock_solver:

                mock_heuristic.return_value = mock_heuristic_result
                mock_solver.return_value = mock_solver_result

                # Act
                import time
                start_time = time.time()

                result = dispatch_run(
                    company_id=mock_company.id,
                    mode="auto",
                    regular_first=True,
                    allow_emergency=True,
                    settings=default_settings
                )

                execution_time = time.time() - start_time

                # Assert
                assert result is not None
                assert result["assignments_count"] == 40
                assert result["unassigned_count"] == 10
                assert execution_time < 5.0  # Doit s'exécuter en moins de 5 secondes

                # Vérifier les métriques de performance
                assert "heuristic_time_ms" in result
                assert "solver_time_ms" in result
                assert "osrm_calls" in result
                assert result["heuristic_time_ms"] == 3000
                assert result["solver_time_ms"] == 1000
                assert result["osrm_calls"] == 200


if __name__ == "__main__":
    pytest.main([__file__])
