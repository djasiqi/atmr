"""Interface abstraite pour le service de routing.

Cette interface permet de :
1. Faciliter les tests (mocks)
2. Préparer une migration vers microservices (remplacement par API REST)
3. Améliorer la séparation des responsabilités
"""
# pyright: reportImplicitOverride=false
# Note: Les méthodes de RoutingServiceLocal utilisent @override mais basedpyright
# ne le reconnaît pas toujours dans ce contexte (problème connu avec les imports conditionnels)

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

# Import override - typing_extensions est garanti disponible (dans requirements.base.txt)
try:
    from typing import (
        override,
    )
except ImportError:
    from typing_extensions import override  # Python < 3.12


class RoutingServiceInterface(ABC):
    """Interface pour le service de routing."""

    @abstractmethod
    def build_distance_matrix(
        self,
        coords: List[Tuple[float, float]],
        *,
        base_url: str | None = None,
        profile: str = "driving",
        timeout: int | None = None,
        max_sources_per_call: int = 60,
        rate_limit_per_sec: int = 8,
        max_retries: int = 2,
        backoff_ms: int = 250,
        redis_client: Any | None = None,
        coord_precision: int = 5,
        avg_speed_kmh_fallback: float = 50.0,
    ) -> List[List[float]]:
        """Construit une matrice de distances en secondes.

        Args:
            coords: Liste de coordonnées (lat, lon)
            base_url: URL OSRM (optionnel, fallback env OSRM_BASE_URL)
            profile: Profil de routing ("driving", "walking", "cycling")
            timeout: Timeout en secondes

        Returns:
            Matrice NxN de durées en secondes (float)
        """
        pass

    @abstractmethod
    def eta_seconds(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        *,
        base_url: str | None = None,
        profile: str = "driving",
        waypoints: List[Tuple[float, float]] | None = None,
        timeout: int = 10,
        redis_client: Any | None = None,
        coord_precision: int = 5,
        avg_speed_kmh_fallback: float = 50.0,
    ) -> int:
        """Calcule un ETA (secondes) robuste via le provider de routing."""
        pass

    @abstractmethod
    def get_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        *,
        profile: str = "driving",
        waypoints: List[Tuple[float, float]] | None = None,
    ) -> Dict[str, Any]:
        """Calcule un itinéraire entre deux points.

        Args:
            origin: Point d'origine (lat, lon)
            destination: Point de destination (lat, lon)
            profile: Profil de routing
            waypoints: Points intermédiaires optionnels

        Returns:
            Dict avec informations de l'itinéraire
        """
        pass

    @abstractmethod
    def invalidate_cache(
        self,
        coords: List[Tuple[float, float]] | None = None,
        zone_id: str | None = None,
    ) -> None:
        """Invalide le cache pour des coordonnées ou une zone.

        Args:
            coords: Liste de coordonnées à invalider
            zone_id: ID de zone à invalider
        """
        pass


class RoutingServiceLocal(RoutingServiceInterface):
    """Implémentation locale (monolithique) du service de routing."""

    @override
    def build_distance_matrix(
        self,
        coords: List[Tuple[float, float]],
        *,
        base_url: str | None = None,
        profile: str = "driving",
        timeout: int | None = None,
        max_sources_per_call: int = 60,
        rate_limit_per_sec: int = 8,
        max_retries: int = 2,
        backoff_ms: int = 250,
        redis_client: Any | None = None,
        coord_precision: int = 5,
        avg_speed_kmh_fallback: float = 50.0,
    ) -> List[List[float]]:
        """Implémentation locale via services.osrm_client."""
        from services.geolocation.osrm import build_distance_matrix_osrm_with_cb

        resolved_base_url = base_url or os.getenv("OSRM_BASE_URL", "http://osrm:5000")
        return build_distance_matrix_osrm_with_cb(
            coords,
            base_url=resolved_base_url,
            profile=profile,
            timeout=timeout,
            max_sources_per_call=max_sources_per_call,
            rate_limit_per_sec=rate_limit_per_sec,
            max_retries=max_retries,
            backoff_ms=backoff_ms,
            redis_client=redis_client,
            coord_precision=coord_precision,
            avg_speed_kmh_fallback=avg_speed_kmh_fallback,
        )

    @override
    def eta_seconds(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        *,
        base_url: str | None = None,
        profile: str = "driving",
        waypoints: List[Tuple[float, float]] | None = None,
        timeout: int = 10,
        redis_client: Any | None = None,
        coord_precision: int = 5,
        avg_speed_kmh_fallback: float = 50.0,
    ) -> int:
        """Implémentation locale via services.osrm_client (route_info + cache)."""
        from services.geolocation.osrm import eta_seconds as _eta_seconds

        resolved_base_url = base_url or os.getenv("OSRM_BASE_URL", "http://osrm:5000")
        return _eta_seconds(
            origin,
            destination,
            base_url=resolved_base_url,
            profile=profile,
            waypoints=waypoints,
            timeout=timeout,
            redis_client=redis_client,
            coord_precision=coord_precision,
            avg_speed_kmh_fallback=avg_speed_kmh_fallback,
        )

    @override
    def get_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        *,
        profile: str = "driving",
        waypoints: List[Tuple[float, float]] | None = None,
    ) -> Dict[str, Any]:
        """Implémentation locale via services.osrm_client."""
        from services.geolocation.osrm import (
            get_route as _get_route,
        )

        base_url = os.getenv("OSRM_BASE_URL", "http://osrm:5000")
        return _get_route(
            base_url=base_url,
            profile=profile,
            origin=origin,
            destination=destination,
            waypoints=waypoints,
        )

    @override
    def invalidate_cache(
        self,
        coords: List[Tuple[float, float]] | None = None,
        zone_id: str | None = None,
    ) -> None:
        """Implémentation locale via services.cache_invalidation."""
        from services.cache_invalidation import invalidate_osrm_matrix_cache

        invalidate_osrm_matrix_cache(coords=coords, zone_id=zone_id)


# Instance par défaut (monolithique)
_default_routing_service: RoutingServiceInterface = RoutingServiceLocal()


def get_routing_service() -> RoutingServiceInterface:
    """Récupère l'instance du service de routing.

    Dans une architecture microservices, cette fonction pourrait retourner
    un client HTTP vers le service de routing distant.

    Returns:
        Instance du service de routing
    """
    return _default_routing_service


def set_routing_service(service: RoutingServiceInterface) -> None:
    """Définit l'instance du service de routing (pour tests).

    Args:
        service: Instance du service de routing
    """
    # Mettre à jour via le module pour éviter global statement
    import services.interfaces.routing_interface as module

    module._default_routing_service = service

