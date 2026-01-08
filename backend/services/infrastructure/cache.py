# backend/services/cache_invalidation.py
"""Utilitaires pour invalidation manuelle des caches."""

import hashlib
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Préfixes de clés de cache
CACHE_PREFIX_NOMINATIM = "nominatim:geocode:"
CACHE_PREFIX_GOOGLE = "google:geocode:"
CACHE_PREFIX_OSRM_TABLE = "osrm:table:"
CACHE_PREFIX_OSRM_PRECOMPUTED = "osrm:precomputed:zone:"
CACHE_PREFIX_DISPATCH_STATUS = "dispatch:status:"
CACHE_PREFIX_WEATHER = "weather:"

# Tags pour invalidation groupée
TAG_DISPATCH = "dispatch:"
TAG_BOOKING = "booking:"


def _get_redis_client() -> Any | None:
    """Récupère un client Redis avec fallback.

    Returns:
        Client Redis ou None si indisponible
    """
    try:
        from ext import redis_client as ext_redis_client

        if ext_redis_client is not None:
            ext_redis_client.ping()
            return ext_redis_client
    except Exception:
        pass

    # Fallback : essayer de créer depuis REDIS_URL
    try:
        redis_url = os.getenv("REDIS_URL", None)
        if redis_url:
            import redis  # pyright: ignore[reportMissingImports]

            socket_timeout = int(os.getenv("REDIS_SOCKET_TIMEOUT", "5"))
            socket_connect_timeout = int(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
            client = redis.from_url(
                redis_url,
                decode_responses=False,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
            )
            client.ping()
            return client
    except Exception:
        pass

    return None


def _normalize_address_for_cache(address: str, country: str | None = None) -> str:
    """Normalise une adresse pour la clé de cache.

    Args:
        address: Adresse à normaliser
        country: Code pays optionnel

    Returns:
        Adresse normalisée
    """
    # Normaliser l'adresse (minuscules, espaces multiples)
    normalized = " ".join(address.lower().strip().split())
    if country:
        normalized = f"{normalized}|{country.upper()}"
    return normalized


def invalidate_geocoding_cache(
    address: str, country: str | None = None, provider: str = "both"
) -> None:
    """Invalide le cache de géocodage pour une adresse.

    ✅ P1: Invalidation manuelle du cache géocodage si adresse mise à jour.

    Args:
        address: Adresse à invalider
        country: Code pays optionnel
        provider: "nominatim", "google", ou "both" (défaut)
    """
    redis_client = _get_redis_client()
    if not redis_client:
        logger.debug("[Cache] Redis unavailable, skipping geocoding cache invalidation")
        return

    try:
        # Normaliser l'adresse pour la clé de cache
        cache_key_normalized = _normalize_address_for_cache(address, country)
        cache_key_hash = hashlib.md5(
            cache_key_normalized.encode("utf-8"), usedforsecurity=False
        ).hexdigest()

        # Invalider Nominatim si demandé
        if provider in ("nominatim", "both"):
            nominatim_key = f"{CACHE_PREFIX_NOMINATIM}{cache_key_hash}"
            redis_client.delete(nominatim_key)
            logger.debug(
                "[Cache] Invalidated Nominatim cache for address: %s", address[:50]
            )

        # Invalider Google Maps si demandé
        if provider in ("google", "both"):
            google_key = f"{CACHE_PREFIX_GOOGLE}{cache_key_hash}"
            redis_client.delete(google_key)
            logger.debug(
                "[Cache] Invalidated Google Maps cache for address: %s", address[:50]
            )

        # Invalider aussi le cache local LRU (si accessible)
        # Note: Le cache local LRU est partagé entre threads mais pas entre instances
        # On ne peut pas l'invalider directement, mais il expirera naturellement
        logger.info(
            "[Cache] ✅ Geocoding cache invalidated for address: %s (provider=%s)",
            address[:50],
            provider,
        )
    except Exception as e:
        logger.warning("[Cache] Failed to invalidate geocoding cache: %s", e)


def invalidate_osrm_matrix_cache(
    coords: list[tuple[float, float]] | None = None,
    cache_key: str | None = None,
    zone_id: str | None = None,
) -> None:
    """Invalide le cache de matrices OSRM.

    ✅ P1: Invalidation manuelle du cache OSRM si coordonnées changent.

    Args:
        coords: Liste de coordonnées (lat, lon) pour calculer la clé de cache
        cache_key: Clé de cache explicite (alternative à coords)
        zone_id: ID de zone pour matrices pré-calculées (format: "lat,lon")
    """
    redis_client = _get_redis_client()
    if not redis_client:
        logger.debug("[Cache] Redis unavailable, skipping OSRM cache invalidation")
        return

    try:
        # Invalider matrice pré-calculée si zone_id fourni
        if zone_id:
            for profile in ["driving", "walking", "cycling"]:
                precomputed_key = f"{CACHE_PREFIX_OSRM_PRECOMPUTED}{zone_id}:{profile}"
                redis_client.delete(precomputed_key)
                logger.debug(
                    "[Cache] Invalidated precomputed OSRM matrix for zone: %s (profile=%s)",
                    zone_id,
                    profile,
                )

        # Invalider matrice spécifique si cache_key fourni
        if cache_key:
            table_key = f"{CACHE_PREFIX_OSRM_TABLE}{cache_key}"
            redis_client.delete(table_key)
            logger.debug(
                "[Cache] Invalidated OSRM table cache for key: %s", cache_key[:50]
            )

        # Invalider matrice basée sur coords si fourni
        if coords:
            # Calculer la clé de cache comme dans osrm_client.py
            from services.geolocation.osrm import _canonical_key_table

            try:
                n = len(coords)
                all_dests = list(range(n))
                calculated_key = _canonical_key_table(coords, list(range(n)), all_dests)
                table_key = f"{CACHE_PREFIX_OSRM_TABLE}{calculated_key}"
                redis_client.delete(table_key)
                logger.debug(
                    "[Cache] Invalidated OSRM table cache for coords (n=%d)",
                    len(coords),
                )
            except Exception as e:
                logger.warning(
                    "[Cache] Failed to calculate OSRM cache key from coords: %s", e
                )

        logger.info("[Cache] ✅ OSRM matrix cache invalidated")
    except Exception as e:
        logger.warning("[Cache] Failed to invalidate OSRM cache: %s", e)


def invalidate_dispatch_status_cache(
    company_id: int, for_date: str | None = None
) -> None:
    """Invalide le cache de statut dispatch.

    ✅ P1: Invalidation manuelle du cache statut dispatch.

    Args:
        company_id: ID de l'entreprise
        for_date: Date optionnelle (YYYY-MM-DD)
    """
    redis_client = _get_redis_client()
    if not redis_client:
        logger.debug(
            "[Cache] Redis unavailable, skipping dispatch status cache invalidation"
        )
        return

    try:
        if for_date:
            # Invalider cache spécifique pour cette date
            cache_key = f"{CACHE_PREFIX_DISPATCH_STATUS}{company_id}:{for_date}"
            redis_client.delete(cache_key)
            logger.debug(
                "[Cache] Invalidated dispatch status cache for company=%s date=%s",
                company_id,
                for_date,
            )
        else:
            # Invalider tous les caches pour cette entreprise (pattern matching)
            pattern = f"{CACHE_PREFIX_DISPATCH_STATUS}{company_id}:*"
            keys = redis_client.keys(pattern)
            if keys:
                redis_client.delete(*keys)
                logger.debug(
                    "[Cache] Invalidated %d dispatch status cache entries for company=%s",
                    len(keys),
                    company_id,
                )

        logger.info(
            "[Cache] ✅ Dispatch status cache invalidated for company=%s date=%s",
            company_id,
            for_date or "all",
        )
    except Exception as e:
        logger.warning("[Cache] Failed to invalidate dispatch status cache: %s", e)


def invalidate_by_tag(tag: str) -> None:
    """Invalide tous les caches associés à un tag.

    ✅ P1: Invalidation par tag pour invalidation groupée.

    Args:
        tag: Tag à invalider (ex: "dispatch:123", "booking:456")
    """
    redis_client = _get_redis_client()
    if not redis_client:
        logger.debug("[Cache] Redis unavailable, skipping tag-based cache invalidation")
        return

    try:
        # Utiliser Redis SCAN pour trouver toutes les clés avec ce tag
        # Format: tag:{tag}:{cache_key}
        pattern = f"tag:{tag}:*"
        keys = []
        cursor = 0

        while True:
            cursor, batch = redis_client.scan(cursor, match=pattern, count=100)
            keys.extend(batch)
            if cursor == 0:
                break

        if keys:
            # Supprimer toutes les clés trouvées
            redis_client.delete(*keys)
            logger.info(
                "[Cache] ✅ Invalidated %d cache entries for tag: %s", len(keys), tag
            )
        else:
            logger.debug("[Cache] No cache entries found for tag: %s", tag)
    except Exception as e:
        logger.warning("[Cache] Failed to invalidate cache by tag: %s", e)


def set_cache_with_tag(
    cache_key: str, value: Any, ttl: int, tag: str | None = None
) -> None:
    """Définit une valeur dans le cache avec un tag optionnel.

    ✅ P1: Permet d'associer un tag à une clé de cache pour invalidation groupée.

    Args:
        cache_key: Clé de cache
        value: Valeur à stocker
        ttl: TTL en secondes
        tag: Tag optionnel pour invalidation groupée
    """
    redis_client = _get_redis_client()
    if not redis_client:
        logger.debug("[Cache] Redis unavailable, skipping cache set with tag")
        return

    try:
        # Stocker la valeur
        value_str = json.dumps(value) if isinstance(value, (dict, list)) else str(value)

        redis_client.setex(cache_key, ttl, value_str)

        # Si tag fourni, créer une clé de tag pour invalidation groupée
        if tag:
            tag_key = f"tag:{tag}:{cache_key}"
            redis_client.setex(tag_key, ttl, "1")  # Valeur dummy, juste pour la clé

        logger.debug(
            "[Cache] Set cache key=%s with tag=%s (ttl=%ds)", cache_key[:50], tag, ttl
        )
    except Exception as e:
        logger.warning("[Cache] Failed to set cache with tag: %s", e)
