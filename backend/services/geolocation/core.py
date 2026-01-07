"""Service de géolocalisation pour centraliser les calculs de distance.

Ce service encapsule les fonctions géographiques de shared/geo_utils.py
pour améliorer la cohérence, faciliter les tests et permettre une future
évolution vers d'autres algorithmes de calcul de distance.
"""

import logging
from typing import Tuple

from shared.geo_utils import (
    haversine_distance,
    haversine_distance_meters,
    haversine_minutes,
    haversine_seconds,
    haversine_tuple,
)

logger = logging.getLogger(__name__)


class GeolocationService:
    """Service centralisé pour les calculs de distance géographique."""

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise le service de géolocalisation.

        Note: Pas de classe parente, donc pas besoin de super().__init__().
        """
        # Les fonctions sont importées depuis shared/geo_utils.py
        # pour éviter la duplication de code
        pass

    def distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcule la distance Haversine entre deux points GPS en kilomètres.

        Args:
            lat1: Latitude du point 1 en degrés décimaux
            lon1: Longitude du point 1 en degrés décimaux
            lat2: Latitude du point 2 en degrés décimaux
            lon2: Longitude du point 2 en degrés décimaux

        Returns:
            Distance en kilomètres (float)
        """
        return haversine_distance(lat1, lon1, lat2, lon2)

    def distance_meters(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calcule la distance Haversine entre deux points GPS en mètres.

        Args:
            lat1: Latitude du point 1 en degrés décimaux
            lon1: Longitude du point 1 en degrés décimaux
            lat2: Latitude du point 2 en degrés décimaux
            lon2: Longitude du point 2 en degrés décimaux

        Returns:
            Distance en mètres (float)
        """
        return haversine_distance_meters(lat1, lon1, lat2, lon2)

    def distance_tuple(
        self, coord1: Tuple[float, float], coord2: Tuple[float, float]
    ) -> float:
        """Calcule la distance Haversine entre deux tuples de coordonnées (lat, lon).

        Args:
            coord1: Tuple (latitude, longitude) du point 1
            coord2: Tuple (latitude, longitude) du point 2

        Returns:
            Distance en kilomètres (float)
        """
        return haversine_tuple(coord1, coord2)

    def travel_time_minutes(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        avg_speed_kmh: float = 40,
    ) -> float:
        """Calcule le temps de trajet estimé en minutes basé sur la distance Haversine.

        Args:
            lat1, lon1: Coordonnées GPS point départ
            lat2, lon2: Coordonnées GPS point arrivée
            avg_speed_kmh: Vitesse moyenne en km/h (défaut: 40 km/h en ville)

        Returns:
            Temps estimé en minutes (float)
        """
        return haversine_minutes(lat1, lon1, lat2, lon2, avg_speed_kmh)

    def travel_time_seconds(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        avg_speed_kmh: float = 40,
    ) -> int:
        """Calcule le temps de trajet estimé en secondes basé sur la distance Haversine.

        Args:
            lat1, lon1: Coordonnées GPS point départ
            lat2, lon2: Coordonnées GPS point arrivée
            avg_speed_kmh: Vitesse moyenne en km/h (défaut: 40 km/h en ville)

        Returns:
            Temps estimé en secondes (int)
        """
        return haversine_seconds(lat1, lon1, lat2, lon2, avg_speed_kmh)


# Instance par défaut pour faciliter l'utilisation
_default_geolocation_service: GeolocationService = GeolocationService()


def get_geolocation_service() -> GeolocationService:
    """Récupère l'instance du service de géolocalisation.

    Returns:
        Instance du service de géolocalisation
    """
    return _default_geolocation_service


def set_geolocation_service(service: GeolocationService) -> None:
    """Définit l'instance du service de géolocalisation (pour tests).

    Args:
        service: Instance du service de géolocalisation
    """
    import services.geolocation_service as module

    module._default_geolocation_service = service
