"""Interface abstraite pour le service de géocodage.

Cette interface permet de :
1. Faciliter les tests (mocks)
2. Préparer une migration vers microservices (remplacement par API REST)
3. Améliorer la séparation des responsabilités
"""
# pyright: reportImplicitOverride=false
# Note: Les méthodes de GeocodingServiceLocal utilisent @override mais basedpyright
# ne le reconnaît pas toujours dans ce contexte (problème connu avec les imports conditionnels)

from abc import ABC, abstractmethod
from typing import Dict, List

# Import override - typing_extensions est garanti disponible (dans requirements.base.txt)
try:
    from typing import (
        override,
    )
except ImportError:
    from typing_extensions import override  # Python < 3.12


class GeocodingServiceInterface(ABC):
    """Interface pour le service de géocodage."""

    @abstractmethod
    def geocode_address(
        self, address: str, *, country: str | None = None, language: str = "fr"
    ) -> Dict[str, float] | None:
        """Géocode une adresse → {'lat': float, 'lon': float} | None.

        Args:
            address: Adresse à géocoder
            country: Code pays optionnel (ex: "CH")
            language: Langue pour les résultats (défaut: "fr")

        Returns:
            Dict avec 'lat' et 'lon' ou None si échec
        """
        pass

    @abstractmethod
    def geocode_addresses_batch(
        self,
        addresses: List[str],
        *,
        country: str | None = None,
        language: str = "fr",
        provider: str = "auto",
    ) -> Dict[str, Dict[str, float] | None]:
        """Géocode plusieurs adresses en parallèle.

        Args:
            addresses: Liste d'adresses à géocoder
            country: Code pays optionnel
            language: Langue pour les résultats
            provider: "nominatim", "google", ou "auto"

        Returns:
            Dict {adresse: {'lat': float, 'lon': float} | None}
        """
        pass

    @abstractmethod
    def invalidate_cache(self, address: str, country: str | None = None) -> None:
        """Invalide le cache pour une adresse.

        Args:
            address: Adresse à invalider
            country: Code pays optionnel
        """
        pass


class GeocodingServiceLocal(GeocodingServiceInterface):
    """Implémentation locale (monolithique) du service de géocodage."""

    @override
    def geocode_address(
        self, address: str, *, country: str | None = None, language: str = "fr"
    ) -> Dict[str, float] | None:
        """Implémentation locale via services.maps."""
        from services.maps import geocode_address as _geocode_address

        return _geocode_address(address, country=country, language=language)

    @override
    def geocode_addresses_batch(
        self,
        addresses: List[str],
        *,
        country: str | None = None,
        language: str = "fr",
        provider: str = "auto",
    ) -> Dict[str, Dict[str, float] | None]:
        """Implémentation locale via services.maps."""
        from services.maps import geocode_addresses_batch as _geocode_batch

        # geocode_addresses_batch n'accepte pas provider, utilise prefer_google
        prefer_google = provider == "google" or (provider == "auto" and True)
        return _geocode_batch(addresses, country=country, prefer_google=prefer_google)

    @override
    def invalidate_cache(self, address: str, country: str | None = None) -> None:
        """Implémentation locale via services.cache_invalidation."""
        from services.cache_invalidation import invalidate_geocoding_cache

        invalidate_geocoding_cache(address, country=country, provider="both")


# Instance par défaut (monolithique)
_default_geocoding_service: GeocodingServiceInterface = GeocodingServiceLocal()


def get_geocoding_service() -> GeocodingServiceInterface:
    """Récupère l'instance du service de géocodage.

    Dans une architecture microservices, cette fonction pourrait retourner
    un client HTTP vers le service de géocodage distant.

    Returns:
        Instance du service de géocodage
    """
    return _default_geocoding_service


def set_geocoding_service(service: GeocodingServiceInterface) -> None:
    """Définit l'instance du service de géocodage (pour tests).

    Args:
        service: Instance du service de géocodage
    """
    # Utiliser un conteneur pour éviter global statement
    _geocoding_service_container = {"service": _default_geocoding_service}
    _geocoding_service_container["service"] = service
    # Mettre à jour la variable globale via le conteneur
    import services.interfaces.geocoding_interface as module

    module._default_geocoding_service = service
