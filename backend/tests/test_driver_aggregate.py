"""Tests pour l'agrégat Driver."""

from __future__ import annotations

from datetime import datetime

import pytest

from drivers.domain.driver import Driver
from drivers.domain.driver_id import DriverId
from drivers.domain.value_objects import DriverLocation, DriverStatus, DriverType

# Constantes pour les tests
TEST_USER_ID = 1
TEST_COMPANY_ID = 1
TEST_LATITUDE = 46.2
TEST_LONGITUDE = 6.15
TEST_ACCURACY = 10.0


class TestDriverAggregate:
    """Tests pour l'agrégat Driver."""

    def test_create_driver(self):
        """Test création d'un driver."""
        driver = Driver(
            id=DriverId(1),
            user_id=TEST_USER_ID,
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=True,
                is_available=True,
                driver_type=DriverType("REGULAR"),
            ),
        )

        assert driver.id.value == 1
        assert driver.user_id == TEST_USER_ID
        assert driver.status.is_active is True
        assert driver.status.is_available is True

    def test_driver_update_location(self):
        """Test mise à jour de la localisation."""
        driver = Driver(
            id=DriverId(1),
            user_id=TEST_USER_ID,
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=True,
                is_available=True,
                driver_type=DriverType("REGULAR"),
            ),
        )

        driver.update_location(
            latitude=TEST_LATITUDE,
            longitude=TEST_LONGITUDE,
            accuracy=TEST_ACCURACY,
        )

        assert driver.location is not None
        assert driver.location.latitude == TEST_LATITUDE
        assert driver.location.longitude == TEST_LONGITUDE
        assert driver.location.accuracy == TEST_ACCURACY

    def test_driver_update_location_invalid(self):
        """Test que la mise à jour échoue avec des coordonnées invalides."""
        driver = Driver(
            id=DriverId(1),
            user_id=TEST_USER_ID,
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=True,
                is_available=True,
                driver_type=DriverType("REGULAR"),
            ),
        )

        with pytest.raises(ValueError, match="Invalid location"):
            driver.update_location(latitude=200.0, longitude=6.15, accuracy=10.0)

    def test_driver_set_available(self):
        """Test marquer le driver comme disponible."""
        driver = Driver(
            id=DriverId(1),
            user_id=TEST_USER_ID,
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=True,
                is_available=False,
                driver_type=DriverType("REGULAR"),
            ),
        )

        driver.set_available()
        assert driver.status.is_available is True

    def test_driver_set_available_inactive(self):
        """Test qu'on ne peut pas marquer un driver inactif comme disponible."""
        driver = Driver(
            id=DriverId(1),
            user_id=TEST_USER_ID,
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=False,
                is_available=False,
                driver_type=DriverType("REGULAR"),
            ),
        )

        with pytest.raises(ValueError, match="Cannot set available"):
            driver.set_available()

    def test_driver_set_unavailable(self):
        """Test marquer le driver comme indisponible."""
        driver = Driver(
            id=DriverId(1),
            user_id=TEST_USER_ID,
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=True,
                is_available=True,
                driver_type=DriverType("REGULAR"),
            ),
        )

        driver.set_unavailable()
        assert driver.status.is_available is False

    def test_driver_activate(self):
        """Test activation d'un driver."""
        driver = Driver(
            id=DriverId(1),
            user_id=TEST_USER_ID,
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=False,
                is_available=False,
                driver_type=DriverType("REGULAR"),
            ),
        )

        driver.activate()
        assert driver.status.is_active is True

    def test_driver_activate_invalid_user_id(self):
        """Test qu'on ne peut pas activer un driver sans user_id valide."""
        driver = Driver(
            id=DriverId(1),
            user_id=0,  # Invalid
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=False,
                is_available=False,
                driver_type=DriverType("REGULAR"),
            ),
        )

        with pytest.raises(ValueError, match="user_id is required"):
            driver.activate()

    def test_driver_deactivate(self):
        """Test désactivation d'un driver."""
        driver = Driver(
            id=DriverId(1),
            user_id=TEST_USER_ID,
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=True,
                is_available=True,
                driver_type=DriverType("REGULAR"),
            ),
        )

        driver.deactivate()
        assert driver.status.is_active is False
        assert driver.status.is_available is False

    def test_driver_validate(self):
        """Test validation des invariants."""
        driver = Driver(
            id=DriverId(1),
            user_id=TEST_USER_ID,
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=True,
                is_available=True,
                driver_type=DriverType("REGULAR"),
            ),
        )

        assert driver.validate() is True

    def test_driver_validate_invalid_active_no_user_id(self):
        """Test validation échoue si actif sans user_id valide."""
        driver = Driver(
            id=DriverId(1),
            user_id=0,  # Invalid
            company_id=TEST_COMPANY_ID,
            status=DriverStatus(
                is_active=True,  # Active mais user_id invalide
                is_available=True,
                driver_type=DriverType("REGULAR"),
            ),
        )

        assert driver.validate() is False

    def test_driver_location_distance_to(self):
        """Test calcul de distance entre deux localisations."""
        location1 = DriverLocation(
            latitude=46.2,
            longitude=6.15,
            accuracy=10.0,
            timestamp=datetime.now(),
        )
        location2 = DriverLocation(
            latitude=46.3,
            longitude=6.16,
            accuracy=10.0,
            timestamp=datetime.now(),
        )

        distance = location1.distance_to(location2)
        assert distance > 0
        assert distance < 20  # Environ 11 km entre ces deux points

    def test_driver_location_is_stationary(self):
        """Test détection si le driver est stationnaire."""
        location_moving = DriverLocation(
            latitude=46.2,
            longitude=6.15,
            accuracy=10.0,
            timestamp=datetime.now(),
            speed=50.0,  # En mouvement
        )
        location_stationary = DriverLocation(
            latitude=46.2,
            longitude=6.15,
            accuracy=10.0,
            timestamp=datetime.now(),
            speed=0.1,  # Stationnaire
        )

        assert location_moving.is_stationary() is False
        assert location_stationary.is_stationary() is True

    def test_driver_status_can_accept_booking(self):
        """Test vérification si le driver peut accepter une réservation."""
        status_available = DriverStatus(
            is_active=True,
            is_available=True,
            driver_type=DriverType("REGULAR"),
        )
        status_unavailable = DriverStatus(
            is_active=True,
            is_available=False,
            driver_type=DriverType("REGULAR"),
        )
        status_inactive = DriverStatus(
            is_active=False,
            is_available=True,
            driver_type=DriverType("REGULAR"),
        )

        assert status_available.can_accept_booking() is True
        assert status_unavailable.can_accept_booking() is False
        assert status_inactive.can_accept_booking() is False
