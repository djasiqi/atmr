# backend/tests/integration/test_orchestration_performance.py
"""Tests de performance pour les modules d'orchestration.

Mesure les performances (temps d'exécution, mémoire) et valide
qu'il n'y a pas de régression après le refactoring.
"""

from __future__ import annotations  # noqa: I001

import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from tests.factories import BookingFactory, CompanyFactory, DriverFactory
from services.unified_dispatch.orchestration.dispatch_orchestrator import (
    DispatchOrchestrator,
)


class TestExecutionTime:
    """Tests de temps d'exécution."""

    @patch("services.unified_dispatch.locking.RedisLockManager")
    def test_small_dataset_execution_time(self, mock_lock_manager, db):
        """Test : Temps d'exécution avec petit dataset (10 bookings, 5 drivers)."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        # Créer 10 bookings
        bookings = []
        for i in range(10):
            booking = BookingFactory(company_id=company.id)
            booking.pickup_lat = 46.2 + (i * 0.01)
            booking.pickup_lon = 6.1 + (i * 0.01)
            booking.dropoff_lat = 46.3 + (i * 0.01)
            booking.dropoff_lon = 6.2 + (i * 0.01)
            booking.scheduled_time = datetime(2026, 1, 15, 12 + i, 0, 0, tzinfo=UTC)
            bookings.append(booking)

        # Créer 5 drivers
        drivers = []
        for i in range(5):
            driver = DriverFactory(company_id=company.id)
            driver.current_lat = 46.2 + (i * 0.01)
            driver.current_lon = 6.1 + (i * 0.01)
            drivers.append(driver)

        db.session.add_all(bookings + drivers)
        db.session.commit()

        mock_lock_instance = MagicMock()
        mock_lock_instance.acquire_lock.return_value = True
        mock_lock_manager.return_value = mock_lock_instance

        orchestrator = DispatchOrchestrator()

        start_time = time.time()
        result = orchestrator.execute(
            company_id=company.id,
            for_date="2025-01-14",
            mode="auto",
        )
        execution_time = time.time() - start_time

        # Vérifier que le résultat est valide
        assert result is not None

        # Vérifier que le temps d'exécution est raisonnable (< 30 secondes pour petit dataset)
        assert execution_time < 30.0, (
            f"Temps d'exécution trop long: {execution_time:.2f}s"
        )

    @patch("services.unified_dispatch.locking.RedisLockManager")
    def test_medium_dataset_execution_time(self, mock_lock_manager, db):
        """Test : Temps d'exécution avec dataset moyen (100 bookings, 20 drivers)."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        # Créer 100 bookings
        bookings = []
        for i in range(100):
            booking = BookingFactory(company_id=company.id)
            booking.pickup_lat = 46.2 + ((i % 10) * 0.01)
            booking.pickup_lon = 6.1 + ((i % 10) * 0.01)
            booking.dropoff_lat = 46.3 + ((i % 10) * 0.01)
            booking.dropoff_lon = 6.2 + ((i % 10) * 0.01)
            booking.scheduled_time = datetime(
                2026, 1, 15, 12 + (i // 10), 0, 0, tzinfo=UTC
            )
            bookings.append(booking)

        # Créer 20 drivers
        drivers = []
        for i in range(20):
            driver = DriverFactory(company_id=company.id)
            driver.current_lat = 46.2 + (i * 0.01)
            driver.current_lon = 6.1 + (i * 0.01)
            drivers.append(driver)

        db.session.add_all(bookings + drivers)
        db.session.commit()

        mock_lock_instance = MagicMock()
        mock_lock_instance.acquire_lock.return_value = True
        mock_lock_manager.return_value = mock_lock_instance

        orchestrator = DispatchOrchestrator()

        start_time = time.time()
        result = orchestrator.execute(
            company_id=company.id,
            for_date="2025-01-14",
            mode="auto",
        )
        execution_time = time.time() - start_time

        # Vérifier que le résultat est valide
        assert result is not None

        # Vérifier que le temps d'exécution est raisonnable (< 120 secondes pour dataset moyen)
        assert execution_time < 120.0, (
            f"Temps d'exécution trop long: {execution_time:.2f}s"
        )

    @pytest.mark.slow
    @patch("services.unified_dispatch.locking.RedisLockManager")
    def test_large_dataset_execution_time(self, mock_lock_manager, db):
        """Test : Temps d'exécution avec grand dataset (1000 bookings, 50 drivers).

        Marqué comme slow car peut prendre du temps.
        """
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        # Créer 1000 bookings (par batch pour éviter la lenteur)
        bookings = []
        for i in range(1000):
            booking = BookingFactory(company_id=company.id)
            booking.pickup_lat = 46.2 + ((i % 20) * 0.01)
            booking.pickup_lon = 6.1 + ((i % 20) * 0.01)
            booking.dropoff_lat = 46.3 + ((i % 20) * 0.01)
            booking.dropoff_lon = 6.2 + ((i % 20) * 0.01)
            booking.scheduled_time = datetime(
                2026, 1, 15, 12 + ((i // 50) % 12), 0, 0, tzinfo=UTC
            )
            bookings.append(booking)

        # Créer 50 drivers
        drivers = []
        for i in range(50):
            driver = DriverFactory(company_id=company.id)
            driver.current_lat = 46.2 + (i * 0.01)
            driver.current_lon = 6.1 + (i * 0.01)
            drivers.append(driver)

        db.session.add_all(bookings + drivers)
        db.session.commit()

        mock_lock_instance = MagicMock()
        mock_lock_instance.acquire_lock.return_value = True
        mock_lock_manager.return_value = mock_lock_instance

        orchestrator = DispatchOrchestrator()

        start_time = time.time()
        result = orchestrator.execute(
            company_id=company.id,
            for_date="2025-01-14",
            mode="auto",
        )
        execution_time = time.time() - start_time

        # Vérifier que le résultat est valide
        assert result is not None

        # Vérifier que le temps d'exécution est raisonnable (< 600 secondes pour grand dataset)
        assert execution_time < 600.0, (
            f"Temps d'exécution trop long: {execution_time:.2f}s"
        )


class TestMemoryUsage:
    """Tests d'utilisation mémoire."""

    @patch("services.unified_dispatch.locking.RedisLockManager")
    def test_memory_usage_small_dataset(self, mock_lock_manager, db):
        """Test : Utilisation mémoire avec petit dataset."""
        try:
            import tracemalloc
        except ImportError:
            pytest.skip("tracemalloc non disponible")

        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        mock_lock_instance = MagicMock()
        mock_lock_instance.acquire_lock.return_value = True
        mock_lock_manager.return_value = mock_lock_instance

        orchestrator = DispatchOrchestrator()

        # Démarrer le traçage mémoire
        tracemalloc.start()

        result = orchestrator.execute(
            company_id=company.id,
            for_date="2025-01-14",
            mode="auto",
        )

        # Obtenir les statistiques mémoire
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Vérifier que le résultat est valide
        assert result is not None

        # Vérifier que l'utilisation mémoire est raisonnable (< 500 MB pour petit dataset)
        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 500.0, f"Utilisation mémoire trop élevée: {peak_mb:.2f} MB"


class TestPerformanceRegression:
    """Tests de régression de performance."""

    @patch("services.unified_dispatch.locking.RedisLockManager")
    def test_no_performance_regression(self, mock_lock_manager, db):
        """Test : Pas de régression de performance significative."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        mock_lock_instance = MagicMock()
        mock_lock_instance.acquire_lock.return_value = True
        mock_lock_manager.return_value = mock_lock_instance

        orchestrator = DispatchOrchestrator()

        # Mesurer le temps d'exécution
        start_time = time.time()
        result = orchestrator.execute(
            company_id=company.id,
            for_date="2025-01-14",
            mode="auto",
        )
        execution_time = time.time() - start_time

        # Vérifier que le résultat est valide
        assert result is not None

        # Critère : temps d'exécution ne doit pas augmenter de plus de 10%
        # (baseline estimée à 5 secondes pour un dataset minimal)
        baseline_time = 5.0
        max_allowed_time = baseline_time * 1.10

        assert execution_time < max_allowed_time, (
            f"Régression de performance détectée: {execution_time:.2f}s > {max_allowed_time:.2f}s "
            f"(augmentation de {((execution_time / baseline_time - 1) * 100):.1f}%)"
        )
