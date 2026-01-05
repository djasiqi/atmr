# backend/tests/services/unified_dispatch/orchestration/test_clustering_manager.py
"""Tests unitaires pour ClusteringManager.

Tests pour :
- should_use_clustering : Décision d'utiliser le clustering
- create_zones : Création de zones géographiques
- dispatch_zones : Dispatch des zones et fusion des résultats
"""

from __future__ import annotations  # noqa: I001

import pytest
from unittest.mock import MagicMock, patch

from factories import CompanyFactory
from services.unified_dispatch.orchestration.clustering_manager import (
    ClusteringManager,
)


class TestShouldUseClustering:
    """Tests pour la méthode should_use_clustering."""

    def test_clustering_enabled_and_bookings_above_threshold(self):
        """Test : Clustering activé et nombre de bookings > seuil → True."""
        settings = MagicMock()
        settings.features.enable_clustering = True
        settings.clustering.bookings_threshold = 100

        problem = {"bookings": [MagicMock()] * 150}  # 150 bookings > 100

        manager = ClusteringManager()
        result = manager.should_use_clustering(problem, settings)

        assert result is True

    def test_clustering_enabled_but_bookings_below_threshold(self):
        """Test : Clustering activé mais nombre de bookings <= seuil → False."""
        settings = MagicMock()
        settings.features.enable_clustering = True
        settings.clustering.bookings_threshold = 100

        problem = {"bookings": [MagicMock()] * 50}  # 50 bookings <= 100

        manager = ClusteringManager()
        result = manager.should_use_clustering(problem, settings)

        assert result is False

    def test_clustering_disabled(self):
        """Test : Clustering désactivé → False."""
        settings = MagicMock()
        settings.features.enable_clustering = False

        problem = {"bookings": [MagicMock()] * 150}  # Même avec beaucoup de bookings

        manager = ClusteringManager()
        result = manager.should_use_clustering(problem, settings)

        assert result is False

    def test_custom_threshold(self):
        """Test : Seuil personnalisé dans settings."""
        settings = MagicMock()
        settings.features.enable_clustering = True
        settings.clustering.bookings_threshold = 200  # Seuil personnalisé

        problem = {"bookings": [MagicMock()] * 150}  # 150 bookings < 200

        manager = ClusteringManager()
        result = manager.should_use_clustering(problem, settings)

        assert result is False

        # Avec 250 bookings > 200
        problem = {"bookings": [MagicMock()] * 250}
        result = manager.should_use_clustering(problem, settings)
        assert result is True

    def test_default_threshold(self):
        """Test : Seuil par défaut (100) si non configuré."""
        settings = MagicMock()
        settings.features.enable_clustering = True
        # Pas de bookings_threshold configuré
        delattr(settings.clustering, "bookings_threshold")

        problem = {"bookings": [MagicMock()] * 150}  # 150 bookings > 100 (défaut)

        manager = ClusteringManager()
        result = manager.should_use_clustering(problem, settings)

        assert result is True


class TestCreateZones:
    """Tests pour la méthode create_zones."""

    @patch(
        "services.unified_dispatch.orchestration.clustering_manager.GeographicClustering"
    )
    def test_create_zones_success(self, mock_clustering_class, db):
        """Test : Création réussie de zones."""
        settings = MagicMock()
        settings.clustering.max_bookings_per_zone = 50
        settings.clustering.cross_zone_tolerance = 0.15

        mock_bookings = [MagicMock()] * 100
        mock_drivers = [MagicMock()] * 10
        problem = {"bookings": mock_bookings, "drivers": mock_drivers}

        mock_zones = [MagicMock(), MagicMock()]  # 2 zones créées
        mock_clustering_instance = MagicMock()
        mock_clustering_instance.create_zones.return_value = mock_zones
        mock_clustering_class.return_value = mock_clustering_instance

        manager = ClusteringManager()
        zones = manager.create_zones(problem, settings)

        assert zones == mock_zones
        mock_clustering_class.assert_called_once_with(max_bookings_per_zone=50)
        mock_clustering_instance.create_zones.assert_called_once_with(
            bookings=mock_bookings,
            drivers=mock_drivers,
            cross_zone_tolerance=0.15,
        )

    @patch(
        "services.unified_dispatch.orchestration.clustering_manager.GeographicClustering"
    )
    def test_create_zones_default_parameters(self, mock_clustering_class):
        """Test : Paramètres par défaut."""
        settings = MagicMock()
        # Pas de paramètres configurés
        delattr(settings.clustering, "max_bookings_per_zone")
        delattr(settings.clustering, "cross_zone_tolerance")

        problem = {"bookings": [MagicMock()] * 100, "drivers": [MagicMock()] * 10}

        mock_zones = [MagicMock()]
        mock_clustering_instance = MagicMock()
        mock_clustering_instance.create_zones.return_value = mock_zones
        mock_clustering_class.return_value = mock_clustering_instance

        manager = ClusteringManager()
        zones = manager.create_zones(problem, settings)

        assert zones == mock_zones
        mock_clustering_class.assert_called_once_with(max_bookings_per_zone=100)
        mock_clustering_instance.create_zones.assert_called_once_with(
            bookings=problem["bookings"],
            drivers=problem["drivers"],
            cross_zone_tolerance=0.1,
        )

    @patch(
        "services.unified_dispatch.orchestration.clustering_manager.GeographicClustering"
    )
    def test_create_zones_handles_validation_error(self, mock_clustering_class):
        """Test : Gestion des erreurs de validation."""
        settings = MagicMock()
        settings.clustering.max_bookings_per_zone = 50
        settings.clustering.cross_zone_tolerance = 0.15

        problem = {"bookings": [MagicMock()] * 100, "drivers": [MagicMock()] * 10}

        mock_clustering_instance = MagicMock()
        mock_clustering_instance.create_zones.side_effect = ValueError("Invalid data")
        mock_clustering_class.return_value = mock_clustering_instance

        manager = ClusteringManager()
        with pytest.raises(ValueError, match="Invalid data"):
            manager.create_zones(problem, settings)


class TestDispatchZones:
    """Tests pour la méthode dispatch_zones."""

    @patch(
        "services.unified_dispatch.orchestration.clustering_manager.GeographicClustering"
    )
    def test_dispatch_zones_success_multiple_zones(self, mock_clustering_class, db):
        """Test : Dispatch réussi avec plusieurs zones."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        settings = MagicMock()
        settings.clustering.max_bookings_per_zone = 50

        zones = [MagicMock(), MagicMock(), MagicMock()]  # 3 zones
        problem = {"bookings": [MagicMock()] * 100, "drivers": [MagicMock()] * 10}

        mock_assignment1 = MagicMock()
        mock_assignment1.booking_id = 1
        mock_assignment2 = MagicMock()
        mock_assignment2.booking_id = 2

        mock_clustering_result = {
            "assignments": [mock_assignment1, mock_assignment2],
            "unassigned": [3, 4],
        }

        mock_clustering_instance = MagicMock()
        mock_clustering_instance.dispatch_zones.return_value = mock_clustering_result
        mock_clustering_class.return_value = mock_clustering_instance

        manager = ClusteringManager()
        result = manager.dispatch_zones(
            zones=zones,
            company=company,
            problem=problem,
            mode="auto",
            settings=settings,
        )

        assert result["assignments"] == [mock_assignment1, mock_assignment2]
        assert result["unassigned"] == [3, 4]
        assert result["meta"]["zones_count"] == 3
        assert result["meta"]["assignments_count"] == 2
        assert result["meta"]["unassigned_count"] == 2

        mock_clustering_class.assert_called_once_with(max_bookings_per_zone=50)
        mock_clustering_instance.dispatch_zones.assert_called_once_with(
            zones=zones,
            company=company,
            problem=problem,
            mode="auto",
            settings=settings,
        )

    def test_dispatch_zones_single_zone_does_not_dispatch(self, db):
        """Test : Dispatch avec une seule zone (ne dispatche pas)."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        settings = MagicMock()
        zones = [MagicMock()]  # Une seule zone

        manager = ClusteringManager()
        result = manager.dispatch_zones(
            zones=zones,
            company=company,
            problem={"bookings": [], "drivers": []},
            mode="auto",
            settings=settings,
        )

        assert result["assignments"] == []
        assert result["unassigned"] == []
        assert result["meta"]["zones_count"] == 1

    @patch(
        "services.unified_dispatch.orchestration.clustering_manager.GeographicClustering"
    )
    def test_dispatch_zones_handles_validation_error(self, mock_clustering_class, db):
        """Test : Gestion des erreurs de validation."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        settings = MagicMock()
        settings.clustering.max_bookings_per_zone = 50

        zones = [MagicMock(), MagicMock()]

        mock_clustering_instance = MagicMock()
        mock_clustering_instance.dispatch_zones.side_effect = ValueError("Invalid data")
        mock_clustering_class.return_value = mock_clustering_instance

        manager = ClusteringManager()
        with pytest.raises(ValueError, match="Invalid data"):
            manager.dispatch_zones(
                zones=zones,
                company=company,
                problem={"bookings": [], "drivers": []},
                mode="auto",
                settings=settings,
            )

    @patch(
        "services.unified_dispatch.orchestration.clustering_manager.GeographicClustering"
    )
    def test_dispatch_zones_handles_unexpected_error(self, mock_clustering_class, db):
        """Test : Gestion des erreurs inattendues."""
        company = CompanyFactory()
        db.session.add(company)
        db.session.commit()

        settings = MagicMock()
        settings.clustering.max_bookings_per_zone = 50

        zones = [MagicMock(), MagicMock()]

        mock_clustering_instance = MagicMock()
        mock_clustering_instance.dispatch_zones.side_effect = RuntimeError(
            "Unexpected error"
        )
        mock_clustering_class.return_value = mock_clustering_instance

        manager = ClusteringManager()
        with pytest.raises(RuntimeError):
            manager.dispatch_zones(
                zones=zones,
                company=company,
                problem={"bookings": [], "drivers": []},
                mode="auto",
                settings=settings,
            )
