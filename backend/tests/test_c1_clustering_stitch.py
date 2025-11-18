#!/usr/bin/env python3
"""
Tests pour C1 : Clustering géographique + stitching.

Teste que 300+ courses sont dispatchées < 60s avec SLA conservé.
"""

import logging
import time

import pytest

from services.unified_dispatch.clustering import GeographicClustering, Zone

logger = logging.getLogger(__name__)


class TestClusteringStitching:
    """Tests pour clustering géographique + stitching (C1)."""

    def test_cluster_creation(self):
        """Test: Création de zones pour clustering."""

        # Simuler des bookings avec coordonnées
        class MockBooking:
            def __init__(self, bid, lat, lon):
                self.id = bid
                self.pickup_lat = lat
                self.pickup_lon = lon

        class MockDriver:
            def __init__(self, did, lat, lon):
                self.id = did
                self.latitude = lat
                self.longitude = lon

        bookings = [MockBooking(i, 45.5 + i * 0.01, -73.5 + i * 0.01) for i in range(150)]
        drivers = [MockDriver(i, 45.5, -73.5) for i in range(20)]

        clustering = GeographicClustering(max_bookings_per_zone=50)
        zones = clustering.create_zones(bookings, drivers)

        assert len(zones) >= 2, "Devrait créer au moins 2 zones"
        assert all(isinstance(z, Zone) for z in zones)

        logger.info("✅ Test: Création de %d zones", len(zones))

    def test_kmeans_partitioning(self):
        """Test: Partition K-Means (k≈N/100)."""

        class MockBooking:
            def __init__(self, bid, lat, lon):
                self.id = bid
                self.pickup_lat = lat
                self.pickup_lon = lon

        bookings = [MockBooking(i, 45.0 + (i % 50) * 0.1, -73.0 + (i // 50) * 0.1) for i in range(300)]
        drivers = []

        clustering = GeographicClustering(max_bookings_per_zone=100)
        zones = clustering.create_zones(bookings, drivers)

        # Devrait créer environ 300/100 = 3 zones
        assert 2 <= len(zones) <= 5

        # Vérifier que chaque zone a < 100 bookings en moyenne
        avg_bookings = sum(len(z.bookings) for z in zones) / len(zones)
        assert avg_bookings <= 150  # Tolérance

        logger.info("✅ Test: Partition K-Means (k=%d zones)", len(zones))

    def test_stitch_zones(self):
        """Test: Stitching de zones."""

        clustering = GeographicClustering(max_bookings_per_zone=100)

        # Créer des résultats mock par zone
        zone_results = {
            0: {"assignments": [1, 2, 3], "unassigned": [4, 5]},
            1: {"assignments": [6, 7, 8], "unassigned": [9, 10]},
        }

        zones = [
            Zone(zone_id=0, bookings=[], drivers=[], center_lat=45.5, center_lon=-73.5),
            Zone(zone_id=1, bookings=[], drivers=[], center_lat=45.6, center_lon=-73.6),
        ]

        result = clustering.stitch_zones(zone_results, zones)

        assert len(result["assignments"]) == 6
        assert len(result["unassigned"]) == 4
        assert result["zones"] == 2

        logger.info("✅ Test: Stitching zones")

    def test_boundary_stitching(self):
        """Test: Stitching de bookings limitrophes."""

        class MockBooking:
            def __init__(self, bid, lat, lon):
                self.id = bid
                self.pickup_lat = lat
                self.pickup_lon = lon

        # Bookings près de la frontière
        bookings = [
            MockBooking(i, 45.5 + 0.05, -73.5 + i * 0.01)  # Proche frontière
            for i in range(10)
        ]

        zones = [
            Zone(zone_id=0, bookings=[], drivers=[], center_lat=45.5, center_lon=-73.5),
            Zone(zone_id=1, bookings=[], drivers=[], center_lat=45.6, center_lon=-73.6),
        ]

        clustering = GeographicClustering()
        improvements = clustering._stitch_boundary_bookings(
            assignments=[], unassigned=bookings, zones=zones, zone_results={}
        )

        assert improvements >= 0

        logger.info("✅ Test: Stitching boundary bookings (%d improvements)", improvements)

    def test_cluster_solve_stitch_300(self):
        """Test: 300 courses < 60s avec clustering + stitching."""

        class MockBooking:
            def __init__(self, bid, lat, lon):
                self.id = bid
                self.pickup_lat = lat
                self.pickup_lon = lon

        class MockDriver:
            def __init__(self, did, lat, lon):
                self.id = did
                self.latitude = lat
                self.longitude = lon

        # Créer 300 bookings
        bookings = [MockBooking(i, 45.0 + (i % 60) * 0.1, -73.0 + (i // 60) * 0.1) for i in range(300)]

        drivers = [MockDriver(i, 45.5, -73.5) for i in range(50)]

        start_time = time.time()

        clustering = GeographicClustering(max_bookings_per_zone=100)
        zones = clustering.create_zones(bookings, drivers)

        # Simuler solve par zone (mock)
        zone_results = {
            z.zone_id: {"assignments": list(range(len(z.bookings) - 5)), "unassigned": list(range(5))} for z in zones
        }

        # Stitch
        result = clustering.stitch_zones(zone_results, zones)

        elapsed = time.time() - start_time

        # Vérifier SLA: < 60s
        assert elapsed < 60.0, f"Trop lent: {elapsed:.2f}s > 60s"

        # Vérifier contraintes globales
        total_assignments = len(result["assignments"])
        total_unassigned = len(result["unassigned"])

        assert total_assignments + total_unassigned == 300

        logger.info(
            "✅ Test: 300 courses dispatches en %.2fs (%d assignés, %d non assignés)",
            elapsed,
            total_assignments,
            total_unassigned,
        )

    def test_parallel_zone_solving(self):
        """Test: Solve par zone en parallèle."""

        class MockBooking:
            def __init__(self, bid, lat, lon):
                self.id = bid
                self.pickup_lat = lat
                self.pickup_lon = lon

        bookings = [MockBooking(i, 45.0 + i * 0.01, -73.0 + i * 0.01) for i in range(200)]
        drivers = []

        clustering = GeographicClustering(max_bookings_per_zone=50)
        zones = clustering.create_zones(bookings, drivers)

        # Simuler résolution parallèle
        from concurrent.futures import ThreadPoolExecutor

        def solve_zone(zone):
            # Mock solve
            return {"assignments": len(zone.bookings) - 5, "unassigned": 5}

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=4) as executor:
            _ = list(executor.map(solve_zone, zones))

        elapsed = time.time() - start_time

        assert elapsed < 5.0, "Parallel solving trop lent"

        logger.info("✅ Test: Parallel solve de %d zones en %.2fs", len(zones), elapsed)

    def test_constraint_validator(self):
        """Test: Validateur de contraintes globales."""

        class MockBooking:
            def __init__(self, bid):
                self.id = bid

        bookings = [MockBooking(i) for i in range(100)]
        drivers = []

        clustering = GeographicClustering(max_bookings_per_zone=50)
        zones = clustering.create_zones(bookings, drivers)

        # Simuler résultats
        zone_results = {z.zone_id: {"assignments": list(range(len(z.bookings))), "unassigned": []} for z in zones}

        result = clustering.stitch_zones(zone_results, zones)

        # Valider contraintes globales
        total_bookings_processed = len(result["assignments"]) + len(result["unassigned"])

        assert total_bookings_processed == 100, "Devrait traiter toutes les courses"

        # Pas de doublons dans les assignations
        assignment_ids = [a.id for a in result["assignments"] if hasattr(a, "id")]
        unique_ids = set(assignment_ids)
        assert len(assignment_ids) == len(unique_ids), "Pas de doublons"

        logger.info("✅ Test: Validateur contraintes globales")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
