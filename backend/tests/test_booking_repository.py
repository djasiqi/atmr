"""Tests pour le repository Booking."""

from __future__ import annotations

import pytest

from bookings.infrastructure.repositories.sqlalchemy_booking_repository import (
    SqlAlchemyBookingRepository,
)


class TestSqlAlchemyBookingRepository:
    """Tests pour SqlAlchemyBookingRepository."""

    def test_repository_initialization(self):
        """Test que le repository peut être initialisé."""
        repo = SqlAlchemyBookingRepository()
        assert repo is not None

    def test_find_by_id_not_found(self):
        """Test find_by_id retourne None si booking non trouvé.

        Note: Ce test nécessite un contexte d'application Flask.
        Pour les tests unitaires purs, utiliser des mocks.
        """
        # Ce test nécessite un contexte d'application Flask
        # Il sera mieux adapté comme test d'intégration
        # Pour l'instant, on le skip car il nécessite une configuration Flask complète
        pytest.skip(
            "Test d'intégration nécessitant un contexte Flask - à implémenter dans test_integration.py"
        )

    # Note: Les tests d'intégration avec la DB nécessitent une session de test
    # Ils seront ajoutés dans test_integration.py
