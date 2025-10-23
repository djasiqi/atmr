"""
Utilitaires géographiques pour calculs de distance et coordonnées.
Ce module centralise toutes les fonctions géographiques utilisées
dans l'application pour éviter la duplication de code.
"""
from math import asin, atan2, cos, radians, sin, sqrt
from typing import Tuple


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calcule la distance Haversine entre deux points GPS en kilomètres.
    La formule de Haversine donne la distance orthodromique (plus court chemin)
    entre deux points sur une sphère à partir de leurs coordonnées GPS.
    Args:
        lat1: Latitude du point 1 en degrés décimaux
        lon1: Longitude du point 1 en degrés décimaux
        lat2: Latitude du point 2 en degrés décimaux
        lon2: Longitude du point 2 en degrés décimaux
    Returns:
        Distance en kilomètres (float)
    Exemple:
        >>> # Distance Paris (48.8566, 2.3522) -> Lyon (45.7640, 4.8357)
        >>> distance = haversine_distance(48.8566, 2.3522, 45.7640, 4.8357)
        >>> print(f"{distance:.1f} km")
        392.2 km
    Note:
        - Rayon Terre utilisé : 6371 km (moyenne)
        - Précision : ±0.5% (acceptable pour dispatch)
        - Pour calculs ultra-précis, utiliser Vincenty (plus complexe)
    """
    # Rayon de la Terre en kilomètres
    R = 6371.0

    # Conversion degrés -> radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)

    # Différences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Formule de Haversine
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))

    # Distance
    distance_km = R * c

    return distance_km


def haversine_distance_meters(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calcule la distance Haversine en mètres.

    Args:
        lat1, lon1, lat2, lon2: Coordonnées GPS

    Returns:
        Distance en mètres (float)
    """
    return haversine_distance(lat1, lon1, lat2, lon2) * 1000.0


def haversine_tuple(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calcule la distance Haversine entre deux tuples de coordonnées (lat, lon).

    Args:
        coord1: Tuple (latitude, longitude) du point 1
        coord2: Tuple (latitude, longitude) du point 2

    Returns:
        Distance en kilomètres (float)

    Exemple:
        >>> paris = (48.8566, 2.3522)
        >>> lyon = (45.7640, 4.8357)
        >>> distance = haversine_tuple(paris, lyon)
        >>> print(f"{distance:.1f} km")
        392.2 km
    """
    return haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])


def haversine_minutes(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    avg_speed_kmh: float = 40.0
) -> float:
    """
    Calcule le temps de trajet estimé en minutes basé sur la distance Haversine.

    Args:
        lat1, lon1: Coordonnées GPS point départ
        lat2, lon2: Coordonnées GPS point arrivée
        avg_speed_kmh: Vitesse moyenne en km/h (défaut: 40 km/h en ville)

    Returns:
        Temps estimé en minutes (float)

    Exemple:
        >>> # Paris -> Lyon à 40 km/h moyen
        >>> temps = haversine_minutes(48.8566, 2.3522, 45.7640, 4.8357, 40.0)
        >>> print(f"{temps:.0f} minutes")
        588 minutes (environ 10h)
    """
    distance_km = haversine_distance(lat1, lon1, lat2, lon2)
    if avg_speed_kmh <= 0:
        raise ValueError("avg_speed_kmh doit être positif")
    return (distance_km / avg_speed_kmh) * 60.0


def haversine_seconds(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    avg_speed_kmh: float = 40.0
) -> int:
    """
    Calcule le temps de trajet estimé en secondes basé sur la distance Haversine.

    Args:
        lat1, lon1: Coordonnées GPS point départ
        lat2, lon2: Coordonnées GPS point arrivée
        avg_speed_kmh: Vitesse moyenne en km/h (défaut: 40 km/h en ville)

    Returns:
        Temps estimé en secondes (int)
    """
    return int(haversine_minutes(lat1, lon1, lat2, lon2, avg_speed_kmh) * 60)


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Valide que les coordonnées GPS sont dans les plages correctes.

    Args:
        lat: Latitude en degrés décimaux
        lon: Longitude en degrés décimaux

    Returns:
        True si coordonnées valides, False sinon

    Exemple:
        >>> validate_coordinates(48.8566, 2.3522)  # Paris
        True
        >>> validate_coordinates(91.0, 2.0)  # Invalide (lat > 90)
        False
    """
    return (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0)


def get_bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calcule le bearing (cap/direction) du point 1 vers le point 2.

    Args:
        lat1, lon1: Coordonnées GPS point départ
        lat2, lon2: Coordonnées GPS point arrivée

    Returns:
        Bearing en degrés (0-360), où 0=Nord, 90=Est, 180=Sud, 270=Ouest

    Exemple:
        >>> bearing = get_bearing(48.8566, 2.3522, 51.5074, -0.1278)
        >>> # Paris -> Londres : ~330° (Nord-Ouest)
    """
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    dlon_rad = radians(lon2 - lon1)

    x = sin(dlon_rad) * cos(lat2_rad)
    y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon_rad)

    initial_bearing = atan2(x, y)
    bearing_degrees = (initial_bearing * 180 / 3.14159265359 + 360) % 360

    return bearing_degrees


# Alias pour compatibilité avec ancien code
calculate_distance = haversine_distance
compute_haversine = haversine_distance
_haversine_km = haversine_tuple  # Pour maps.py et osrm_client.py
_haversine_distance = haversine_distance  # Pour heuristics.py
_haversine_seconds = haversine_seconds  # Pour maps.py

