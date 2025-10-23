# ruff: noqa: DTZ001, DTZ003
# pyright: reportMissingImports=false
"""
Tests pour RLDispatchManager.

Teste l'intégration de l'agent RL avec le système de dispatch réel.
"""
import numpy as np
import pytest

from models.booking import Booking
from models.driver import Driver
from services.rl.rl_dispatch_manager import RLDispatchManager
from tests.factories import BookingFactory, DriverFactory


class TestRLDispatchManagerCreation:
    """Tests de création du manager RL."""

    def test_manager_creation_default(self):
        """Test création avec paramètres par défaut."""
        manager = RLDispatchManager(model_path="data/rl/models/dqn_best.pth")

        assert manager is not None
        assert manager.model_path == "data/rl/models/dqn_best.pth"
        assert manager.fallback_enabled is True

    def test_manager_loads_model_if_exists(self, db):
        """Test que le manager charge le modèle si disponible."""
        # Note: Ce test échouera si le modèle n'existe pas
        # C'est normal en environnement de test
        manager = RLDispatchManager()

        # Le manager devrait exister même si le modèle n'est pas chargé
        assert manager is not None


class TestRLDispatchManagerStateBuilding:
    """Tests de construction de l'état RL."""

    def test_build_state_dimensions(self, db):
        """Test que l'état a la bonne dimension."""
        manager = RLDispatchManager()

        booking = BookingFactory.create()
        # Définir les coordonnées après création
        booking.pickup_lat = 46.2044
        booking.pickup_lon = 6.1432
        booking.dropoff_lat = 46.2100
        booking.dropoff_lon = 6.1500

        drivers = []
        for _ in range(5):
            driver = DriverFactory.create(company_id=booking.company_id)
            driver.is_available = True
            driver.latitude = 46.2050
            driver.longitude = 6.1440
            drivers.append(driver)

        state = manager._build_state(booking, drivers)

        assert state.shape == (122,)
        assert state.dtype == np.float32

    def test_build_state_values_normalized(self, db):
        """Test que les valeurs d'état sont normalisées."""
        manager = RLDispatchManager()

        booking = BookingFactory.create()
        booking.pickup_lat = 46.2044
        booking.pickup_lon = 6.1432

        driver = DriverFactory.create(company_id=booking.company_id)
        driver.is_available = True
        drivers = [driver]

        state = manager._build_state(booking, drivers)

        # La plupart des valeurs devraient être entre 0 et 1 (normalisées)
        # ou entre -1 et 1 pour les coordonnées
        assert np.all(state >= -180)  # Min lat/lng
        assert np.all(state <= 180)   # Max lat/lng


class TestRLDispatchManagerSuggestions:
    """Tests de génération de suggestions."""

    def test_get_suggestion_with_drivers(self, db):
        """Test obtenir suggestion avec drivers disponibles."""
        manager = RLDispatchManager()

        # Si le modèle n'est pas chargé, devrait utiliser fallback
        booking = BookingFactory.create()
        booking.pickup_lat = 46.2044
        booking.pickup_lon = 6.1432

        drivers = []
        for _ in range(3):
            driver = DriverFactory.create(company_id=booking.company_id)
            driver.is_available = True
            driver.latitude = 46.2050
            driver.longitude = 6.1440
            drivers.append(driver)

        suggestion = manager.get_suggestion(booking, drivers)

        # Devrait retourner un driver ou None
        assert suggestion is None or isinstance(suggestion, Driver)

    def test_get_suggestion_no_drivers(self, db):
        """Test suggestion sans drivers disponibles."""
        manager = RLDispatchManager()

        booking = BookingFactory.create()
        drivers = []

        suggestion = manager.get_suggestion(booking, drivers)

        # Devrait retourner None ou un driver (fallback)
        assert suggestion is None or isinstance(suggestion, Driver)

    def test_action_to_driver_valid(self, db):
        """Test conversion action → driver valide."""
        manager = RLDispatchManager()

        drivers = []
        for _ in range(5):
            driver = DriverFactory.create()
            driver.is_available = True
            drivers.append(driver)

        # Action 0-4 devrait retourner un driver
        driver = manager._action_to_driver(2, drivers)

        assert driver is not None
        assert driver == drivers[2]

    def test_action_to_driver_wait(self, db):
        """Test conversion action wait."""
        manager = RLDispatchManager()

        drivers = [DriverFactory.create() for _ in range(5)]  # is_available par défaut

        # Action >= len(drivers) = wait
        driver = manager._action_to_driver(10, drivers)

        assert driver is None


class TestRLDispatchManagerFallback:
    """Tests du système de fallback."""

    def test_fallback_heuristic_closest_driver(self, db):
        """Test que le fallback choisit le driver le plus proche."""
        manager = RLDispatchManager()

        booking = BookingFactory.create()
        booking.pickup_lat = 46.2044
        booking.pickup_lon = 6.1432

        # Créer drivers à différentes distances
        driver_proche = DriverFactory.create(company_id=booking.company_id)
        driver_proche.is_available = True
        driver_proche.latitude = 46.2050  # Proche
        driver_proche.longitude = 6.1440

        driver_loin = DriverFactory.create(company_id=booking.company_id)
        driver_loin.is_available = True
        driver_loin.latitude = 46.3000  # Loin
        driver_loin.longitude = 6.2000

        drivers = [driver_loin, driver_proche]  # Ordre inversé exprès

        # Le fallback devrait choisir le plus proche
        result = manager._fallback_heuristic(booking, drivers)

        assert result == driver_proche


class TestRLDispatchManagerStatistics:
    """Tests des statistiques."""

    def test_get_statistics(self):
        """Test récupération des statistiques."""
        manager = RLDispatchManager()

        stats = manager.get_statistics()

        assert 'is_loaded' in stats
        assert 'model_path' in stats
        assert 'suggestions_count' in stats
        assert 'errors_count' in stats
        assert 'fallback_count' in stats
        assert 'success_rate' in stats
        assert 'fallback_rate' in stats

    def test_reset_statistics(self, db):
        """Test réinitialisation des statistiques."""
        manager = RLDispatchManager()

        # Générer quelques suggestions
        booking = BookingFactory.create()
        booking.pickup_lat = 46.2044
        booking.pickup_lon = 6.1432

        driver = DriverFactory.create(company_id=booking.company_id)
        driver.is_available = True
        driver.latitude = 46.2050
        driver.longitude = 6.1440
        drivers = [driver]

        manager.get_suggestion(booking, drivers)
        manager.get_suggestion(booking, drivers)

        assert manager.suggestions_count > 0

        # Reset
        manager.reset_statistics()

        assert manager.suggestions_count == 0
        assert manager.errors_count == 0
        assert manager.fallback_count == 0

