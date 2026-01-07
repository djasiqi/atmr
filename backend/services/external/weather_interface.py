"""Interface abstraite pour le service météo.

Cette interface permet de :
1. Faciliter les tests (mocks)
2. Préparer une migration vers microservices (remplacement par API REST)
3. Améliorer la séparation des responsabilités
"""
# pyright: reportImplicitOverride=false
# Note: Les méthodes de WeatherServiceLocal utilisent @override mais basedpyright
# ne le reconnaît pas toujours dans ce contexte (problème connu avec les imports conditionnels)

from abc import ABC, abstractmethod
from typing import Any, Dict

# Import override - typing_extensions est garanti disponible (dans requirements.base.txt)
try:
    from typing import (
        override,
    )
except ImportError:
    from typing_extensions import override  # Python < 3.12


class WeatherServiceInterface(ABC):
    """Interface pour le service météo."""

    @abstractmethod
    def get_weather(self, lat: float, lon: float) -> Dict[str, Any] | None:
        """Récupère les données météo pour une position.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Dict avec données météo ou None si échec
        """
        pass

    @abstractmethod
    def get_weather_factor(self, lat: float, lon: float) -> float:
        """Calcule un facteur météo (0.0 à 1.0) pour une position.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Facteur météo (0.0 = mauvais, 1.0 = bon)
        """
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        """Vide le cache météo."""
        pass


class WeatherServiceLocal(WeatherServiceInterface):
    """Implémentation locale (monolithique) du service météo."""

    @override
    def get_weather(self, lat: float, lon: float) -> Dict[str, Any] | None:
        """Implémentation locale via services.external.weather."""
        from services.external.weather import WeatherService

        return WeatherService.get_weather(lat, lon)

    @override
    def get_weather_factor(self, lat: float, lon: float) -> float:
        """Implémentation locale via services.external.weather."""
        from services.external.weather import WeatherService

        return WeatherService.get_weather_factor(lat, lon)

    @override
    def clear_cache(self) -> None:
        """Implémentation locale via services.external.weather."""
        from services.external.weather import WeatherService

        WeatherService.clear_cache()


# Instance par défaut (monolithique)
_default_weather_service: WeatherServiceInterface = WeatherServiceLocal()


def get_weather_service() -> WeatherServiceInterface:
    """Récupère l'instance du service météo.

    Dans une architecture microservices, cette fonction pourrait retourner
    un client HTTP vers le service météo distant.

    Returns:
        Instance du service météo
    """
    return _default_weather_service


def set_weather_service(service: WeatherServiceInterface) -> None:
    """Définit l'instance du service météo (pour tests).

    Args:
        service: Instance du service météo
    """
    # Mettre à jour via le module pour éviter global statement
    import services.external.weather_interface as module

    module._default_weather_service = service

