"""Adapter Infrastructure pour l'invalidation du cache (géocodage + OSRM).

Migration progressive vers Clean Architecture:
- Encapsule les appels à `services.infrastructure.cache` depuis la couche Application
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


def invalidate_geocoding_cache_adapter(
    address: str, *, country: str = "CH", provider: str = "both"
) -> None:
    """Adapter pour invalider le cache de géocodage.

    Args:
        address: Adresse à invalider.
        country: Code pays (défaut: CH).
        provider: Provider à invalider (both, google, osm).
    """
    try:
        from services.infrastructure.cache import invalidate_geocoding_cache

        invalidate_geocoding_cache(address, country=country, provider=provider)
        logger.debug(
            "[Cache] ✅ Invalidated geocoding cache for address: %s (country=%s, provider=%s)",
            address[:50],
            country,
            provider,
        )
    except Exception as e:
        logger.warning(
            "[Cache] Failed to invalidate geocoding cache for address %s: %s",
            address[:50],
            e,
        )


def invalidate_osrm_matrix_cache_adapter(coords: list[tuple[float, float]]) -> None:
    """Adapter pour invalider le cache OSRM matrix.

    Args:
        coords: Liste de coordonnées (lat, lon) à invalider.
    """
    try:
        from services.infrastructure.cache import invalidate_osrm_matrix_cache

        invalidate_osrm_matrix_cache(coords=coords)
        logger.debug(
            "[Cache] ✅ Invalidated OSRM matrix cache for %s coordinates",
            len(coords),
        )
    except Exception as e:
        logger.warning(
            "[Cache] Failed to invalidate OSRM matrix cache: %s",
            e,
        )


def create_cache_invalidation_ports() -> tuple[
    Callable[[str], None], Callable[[list[tuple[float, float]]], None]
]:
    """Factory pour créer les ports d'invalidation du cache.

    Returns:
        Tuple (invalidate_geocoding_fn, invalidate_osrm_fn)
    """

    def invalidate_geocoding_fn(address: str) -> None:
        invalidate_geocoding_cache_adapter(address, country="CH", provider="both")

    def invalidate_osrm_fn(coords: list[tuple[float, float]]) -> None:
        invalidate_osrm_matrix_cache_adapter(coords)

    return invalidate_geocoding_fn, invalidate_osrm_fn
