#!/usr/bin/env python3
"""
Tests de fallback OSRM pour l'Étape 15.

Ces tests valident le système de fallback OSRM quand le service principal
n'est pas disponible, garantissant la continuité du service de dispatch.
✅ FIX: Tests simplifiés pour tester le fallback haversine réel
au lieu de classes inexistantes.
"""

import sys
from pathlib import Path
from unittest.mock import patch

# Ajouter le répertoire backend au path Python
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))


class TestOSRMFallback:
    """Tests du système de fallback OSRM avec les fonctions réelles."""

    def test_osrm_build_distance_matrix_osrm_success(self):
        """Test que build_distance_matrix_osrm fonctionne normalement avec mock."""
        from services.geolocation.osrm import build_distance_matrix_osrm

        # Coordonnées de test (Lausanne)
        coords = [(46.2044, 6.1432), (46.2100, 6.1500), (46.2200, 6.1600)]

        # Avec le fixture mock_osrm_client, la fonction devrait retourner une matrice
        result = build_distance_matrix_osrm(
            coords=coords, base_url="http://localhost:5000"
        )

        # Vérifier que le résultat est une matrice carrée
        assert isinstance(result, list)
        assert len(result) == len(coords)
        assert all(isinstance(row, list) and len(row) == len(coords) for row in result)

        # Vérifier que la diagonale est à 0
        for i in range(len(coords)):
            assert result[i][i] == 0.0

        print("  ✅ build_distance_matrix_osrm fonctionne correctement")

    def test_osrm_build_distance_matrix_osrm_fallback_on_error(self):
        """Test que build_distance_matrix_osrm utilise le fallback haversine
        quand OSRM échoue."""
        from services.geolocation.osrm import build_distance_matrix_osrm

        # Coordonnées de test (Lausanne)
        coords = [(46.2044, 6.1432), (46.2100, 6.1500)]

        # Simuler un échec OSRM en patchant la fonction _table pour lever une exception
        with patch("services.osrm_client._table") as mock_table:
            mock_table.side_effect = Exception("OSRM service unavailable")

            # Appeler avec un timeout très court et max_retries=0 pour forcer l'échec
            result = build_distance_matrix_osrm(
                coords=coords,
                base_url="http://localhost:5000",
                timeout=1,
                max_retries=0,
            )

            # Vérifier que le résultat est une matrice (fallback haversine)
            assert isinstance(result, list)
            assert len(result) == len(coords)
            assert all(
                isinstance(row, list) and len(row) == len(coords) for row in result
            )

            # Vérifier que la diagonale est à 0
            for i in range(len(coords)):
                assert result[i][i] == 0.0

            # Vérifier que les durées sont cohérentes
            # (non nulles pour les paires différentes)
            assert result[0][1] > 0.0
            assert result[1][0] > 0.0

        print("  ✅ Fallback haversine fonctionne quand OSRM échoue")

    def test_osrm_route_info_success(self):
        """Test que route_info fonctionne normalement avec mock."""
        from services.geolocation.osrm import route_info

        # Coordonnées de test (Lausanne)
        origin = (46.2044, 6.1432)
        dest = (46.2100, 6.1500)

        # Avec le fixture mock_osrm_client, la fonction devrait retourner
        # des données de route
        result = route_info(
            origin=origin, destination=dest, base_url="http://localhost:5000"
        )

        # Vérifier que le résultat contient les champs attendus
        assert isinstance(result, dict)
        assert "duration" in result
        assert "distance" in result
        assert result["duration"] >= 0
        assert result["distance"] >= 0

        print("  ✅ route_info fonctionne correctement")

    def test_osrm_route_info_fallback_on_error(self):
        """Test que route_info utilise le fallback haversine quand OSRM échoue."""
        from services.geolocation.osrm import (
            _fallback_eta_seconds,
            _haversine_km,
            route_info,
        )

        # Coordonnées de test (Lausanne)
        origin = (46.2044, 6.1432)
        dest = (46.2100, 6.1500)

        # Simuler un échec OSRM
        with patch("services.osrm_client._route") as mock_route:
            mock_route.side_effect = Exception("OSRM service unavailable")

            # Appeler avec un timeout très court pour forcer l'échec
            result = route_info(
                origin=origin,
                destination=dest,
                base_url="http://localhost:5000",
                timeout=1,
            )

            # Vérifier que le résultat contient les champs attendus (fallback)
            assert isinstance(result, dict)
            assert "duration" in result
            assert "distance" in result
            assert result["duration"] >= 0
            assert result["distance"] >= 0

            # Vérifier que les valeurs sont cohérentes avec haversine
            km = _haversine_km(origin, dest)
            expected_duration = _fallback_eta_seconds(origin, dest)
            # Les valeurs peuvent différer légèrement, mais doivent être proches
            assert (
                abs(result["duration"] - expected_duration) < 100
            )  # tolérance de 100s
            assert abs(result["distance"] - km * 1000) < 1000  # tolérance de 1000m

        print("  ✅ Fallback haversine fonctionne pour route_info")

    def test_osrm_fallback_matrix_symmetry(self):
        """Test que la matrice de fallback est symétrique."""
        from services.geolocation.osrm import _fallback_matrix

        # Coordonnées de test
        coords = [(46.2044, 6.1432), (46.2100, 6.1500), (46.2200, 6.1600)]

        matrix = _fallback_matrix(coords)

        # Vérifier la symétrie (durée de A à B = durée de B à A)
        for i in range(len(coords)):
            for j in range(len(coords)):
                if i != j:
                    # Les durées doivent être proches (symétrie approximative)
                    assert abs(matrix[i][j] - matrix[j][i]) < 1.0  # tolérance de 1s

        print("  ✅ Matrice de fallback est symétrique")

    def test_osrm_fallback_matrix_diagonal_zero(self):
        """Test que la diagonale de la matrice de fallback est à zéro."""
        from services.geolocation.osrm import _fallback_matrix

        # Coordonnées de test
        coords = [(46.2044, 6.1432), (46.2100, 6.1500), (46.2200, 6.1600)]

        matrix = _fallback_matrix(coords)

        # Vérifier que la diagonale est à 0
        for i in range(len(coords)):
            assert matrix[i][i] == 0.0

        print("  ✅ Diagonale de la matrice de fallback est à zéro")


class TestOSRMFallbackIntegration:
    """Tests d'intégration du fallback OSRM avec le système de dispatch."""

    def test_osrm_fallback_with_dispatch_data(self):
        """Test que le fallback OSRM fonctionne avec les données du dispatch."""
        from services.geolocation.osrm import build_distance_matrix_osrm

        # Simuler des coordonnées de drivers et bookings
        driver_coords = [(46.2044, 6.1432), (46.2100, 6.1500)]
        booking_coords = [(46.2200, 6.1600), (46.2300, 6.1700)]

        # Combiner toutes les coordonnées pour la matrice
        all_coords = driver_coords + booking_coords

        # Construire la matrice
        matrix = build_distance_matrix_osrm(
            coords=all_coords, base_url="http://localhost:5000"
        )

        # Vérifier que la matrice est correcte
        assert len(matrix) == len(all_coords)
        assert all(len(row) == len(all_coords) for row in matrix)

        # Vérifier que les durées sont positives pour les paires différentes
        for i in range(len(all_coords)):
            for j in range(len(all_coords)):
                if i != j:
                    assert matrix[i][j] >= 0

        print("  ✅ Fallback OSRM fonctionne avec données de dispatch")


if __name__ == "__main__":
    # Exécution des tests
    print("🚀 TESTS DE FALLBACK OSRM")
    print("=" * 50)

    test_instance = TestOSRMFallback()

    # Tests de base
    test_instance.test_osrm_build_distance_matrix_osrm_success()
    test_instance.test_osrm_build_distance_matrix_osrm_fallback_on_error()
    test_instance.test_osrm_route_info_success()
    test_instance.test_osrm_route_info_fallback_on_error()
    test_instance.test_osrm_fallback_matrix_symmetry()
    test_instance.test_osrm_fallback_matrix_diagonal_zero()

    # Tests d'intégration
    integration_instance = TestOSRMFallbackIntegration()
    integration_instance.test_osrm_fallback_with_dispatch_data()

    print("=" * 50)
    print("✅ TOUS LES TESTS DE FALLBACK OSRM RÉUSSIS")

