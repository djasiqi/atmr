"""
Tests unitaires pour shared/geo_utils.py
"""
import pytest

from shared.geo_utils import (
    get_bearing,
    haversine_distance,
    haversine_distance_meters,
    haversine_minutes,
    haversine_seconds,
    haversine_tuple,
    validate_coordinates,
)


class TestHaversineDistance:
    """Tests pour calcul distance Haversine."""

    def test_distance_paris_lyon(self):
        """Distance Paris -> Lyon (~392 km)."""
        distance = haversine_distance(48.8566, 2.3522, 45.7640, 4.8357)
        assert 390 < distance < 395, f"Distance incorrecte: {distance}"

    def test_distance_same_point(self):
        """Distance entre même point = 0."""
        distance = haversine_distance(48.8566, 2.3522, 48.8566, 2.3522)
        assert distance < 0.0001  # Presque 0

    def test_distance_meters(self):
        """Version en mètres."""
        distance_km = haversine_distance(48.8566, 2.3522, 45.7640, 4.8357)
        distance_m = haversine_distance_meters(48.8566, 2.3522, 45.7640, 4.8357)
        assert abs(distance_m - distance_km * 1000) < 1

    def test_distance_geneva_lausanne(self):
        """Distance Genève (46.2044, 6.1432) -> Lausanne (46.5197, 6.6323) ~52 km."""
        distance = haversine_distance(46.2044, 6.1432, 46.5197, 6.6323)
        assert 50 < distance < 55, f"Distance incorrecte: {distance}"

    def test_haversine_tuple(self):
        """Test avec tuples de coordonnées."""
        paris = (48.8566, 2.3522)
        lyon = (45.7640, 4.8357)
        distance = haversine_tuple(paris, lyon)
        assert 390 < distance < 395


class TestHaversineTime:
    """Tests pour calculs de temps."""

    def test_haversine_minutes_default_speed(self):
        """Calcul temps en minutes avec vitesse par défaut (40 km/h)."""
        # Paris -> Lyon ~392 km à 40 km/h = ~588 minutes
        temps = haversine_minutes(48.8566, 2.3522, 45.7640, 4.8357)
        assert 580 < temps < 600

    def test_haversine_minutes_custom_speed(self):
        """Calcul temps en minutes avec vitesse personnalisée."""
        # 10 km à 50 km/h = 12 minutes
        temps = haversine_minutes(48.8566, 2.3522, 48.9566, 2.3522, avg_speed_kmh=50.0)
        assert 10 < temps < 15

    def test_haversine_seconds(self):
        """Calcul temps en secondes."""
        temps_min = haversine_minutes(48.8566, 2.3522, 48.9566, 2.3522, avg_speed_kmh=40.0)
        temps_sec = haversine_seconds(48.8566, 2.3522, 48.9566, 2.3522, avg_speed_kmh=40.0)
        assert abs(temps_sec - temps_min * 60) < 1

    def test_haversine_minutes_invalid_speed(self):
        """Erreur si vitesse négative ou nulle."""
        with pytest.raises(ValueError):
            haversine_minutes(48.8566, 2.3522, 45.7640, 4.8357, avg_speed_kmh=0)

        with pytest.raises(ValueError):
            haversine_minutes(48.8566, 2.3522, 45.7640, 4.8357, avg_speed_kmh=-10)


class TestValidateCoordinates:
    """Tests validation coordonnées."""

    def test_valid_coordinates(self):
        """Coordonnées valides."""
        assert validate_coordinates(48.8566, 2.3522) is True  # Paris
        assert validate_coordinates(0.0, 0.0) is True  # Équateur/Méridien
        assert validate_coordinates(90.0, 180.0) is True  # Limites max
        assert validate_coordinates(-90.0, -180.0) is True  # Limites min

    def test_invalid_latitude(self):
        """Latitude invalide."""
        assert validate_coordinates(91.0, 2.0) is False  # > 90
        assert validate_coordinates(-91.0, 2.0) is False  # < -90
        assert validate_coordinates(100.0, 2.0) is False  # Très hors limites

    def test_invalid_longitude(self):
        """Longitude invalide."""
        assert validate_coordinates(48.0, 181.0) is False  # > 180
        assert validate_coordinates(48.0, -181.0) is False  # < -180
        assert validate_coordinates(48.0, 200.0) is False  # Très hors limites

    def test_edge_cases(self):
        """Cas limites."""
        assert validate_coordinates(90.0, 0.0) is True  # Pôle Nord
        assert validate_coordinates(-90.0, 0.0) is True  # Pôle Sud
        assert validate_coordinates(0.0, 180.0) is True  # Anti-méridien


class TestGetBearing:
    """Tests calcul bearing."""

    def test_bearing_north(self):
        """Bearing vers le Nord (~0°)."""
        bearing = get_bearing(45.0, 6.0, 46.0, 6.0)
        assert 0 <= bearing < 10 or bearing > 350  # Approximativement Nord

    def test_bearing_east(self):
        """Bearing vers l'Est (~90°)."""
        bearing = get_bearing(45.0, 6.0, 45.0, 7.0)
        assert 85 < bearing < 95  # Approximativement Est

    def test_bearing_south(self):
        """Bearing vers le Sud (~180°)."""
        bearing = get_bearing(46.0, 6.0, 45.0, 6.0)
        assert 175 < bearing < 185  # Approximativement Sud

    def test_bearing_west(self):
        """Bearing vers l'Ouest (~270°)."""
        bearing = get_bearing(45.0, 7.0, 45.0, 6.0)
        assert 265 < bearing < 275  # Approximativement Ouest

    def test_bearing_range(self):
        """Le bearing doit toujours être entre 0 et 360."""
        bearing = get_bearing(48.8566, 2.3522, 51.5074, -0.1278)  # Paris -> Londres
        assert 0 <= bearing < 360


class TestAliases:
    """Tests des alias pour compatibilité."""

    def test_calculate_distance_alias(self):
        """Alias calculate_distance."""
        from shared.geo_utils import calculate_distance
        distance = calculate_distance(48.8566, 2.3522, 45.7640, 4.8357)
        assert 390 < distance < 395

    def test_compute_haversine_alias(self):
        """Alias compute_haversine."""
        from shared.geo_utils import compute_haversine
        distance = compute_haversine(48.8566, 2.3522, 45.7640, 4.8357)
        assert 390 < distance < 395

