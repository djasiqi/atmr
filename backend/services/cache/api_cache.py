"""
Cache utilitaire pour les réponses API.

Fournit un cache Redis simple pour éviter les requêtes en doublon
et réduire la charge sur la base de données.
"""

from __future__ import annotations

import hashlib
import json
import logging
from functools import wraps
from typing import Any, Callable

from flask import request

logger = logging.getLogger(__name__)

# TTL par défaut : 5 secondes (suffisant pour éviter les requêtes en doublon)
DEFAULT_CACHE_TTL = 5

# Code HTTP pour les réponses réussies
HTTP_OK = 200


def _get_redis_client():
    """Obtenir le client Redis de manière sécurisée."""
    try:
        from ext import redis_client

        return redis_client if redis_client else None
    except (ImportError, AttributeError):
        logger.debug("[Cache] Redis non disponible")
        return None


def _generate_cache_key(prefix: str, *args: Any, **kwargs: Any) -> str:
    """
    Génère une clé de cache unique basée sur le préfixe et les paramètres.

    Args:
        prefix: Préfixe de la clé (ex: "api:rides")
        *args: Arguments positionnels à inclure dans la clé
        **kwargs: Arguments nommés à inclure dans la clé

    Returns:
        Clé de cache unique
    """
    # Combiner tous les paramètres
    key_parts = [prefix]
    key_parts.extend(str(arg) for arg in args)
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))

    # Générer un hash pour une clé courte et stable
    key_str = ":".join(key_parts)
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]

    return f"cache:{prefix}:{key_hash}"


def cache_response(
    prefix: str,
    ttl: int = DEFAULT_CACHE_TTL,
    key_func: Callable[..., tuple[tuple[Any, ...], dict[str, Any]]] | None = None,
):
    """
    Décorateur pour mettre en cache les réponses API.

    Args:
        prefix: Préfixe pour la clé de cache
        ttl: Durée de vie du cache en secondes
        key_func: Fonction optionnelle pour extraire les paramètres de la clé
                  Doit retourner (args, kwargs) à utiliser pour la clé

    Usage:
        @cache_response("api:rides", ttl=5)
        def get_rides(self):
            # ... logique de récupération ...
            return data, 200

    Notes:
        - Le cache est basé sur les paramètres de requête (request.args)
        - Si Redis n'est pas disponible, le décorateur n'a aucun effet
        - Les réponses avec code != 200 ne sont pas mises en cache
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, int]:
            redis_client = _get_redis_client()

            # Si Redis n'est pas disponible, exécuter normalement
            if not redis_client:
                return func(*args, **kwargs)

            # Générer la clé de cache
            if key_func:
                cache_args, cache_kwargs = key_func(*args, **kwargs)
            else:
                # Par défaut, utiliser les paramètres de requête
                cache_args = ()
                cache_kwargs = dict(request.args)

            cache_key = _generate_cache_key(prefix, *cache_args, **cache_kwargs)

            try:
                # Vérifier si la réponse est en cache
                cached = redis_client.get(cache_key)
                if cached:
                    logger.debug("[Cache] HIT: %s", cache_key)
                    try:
                        cached_data = json.loads(cached)
                        return cached_data["response"], cached_data["status"]
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning("[Cache] Erreur de désérialisation: %s", e)
                        # Continuer avec l'exécution normale

                # Cache MISS : exécuter la fonction
                logger.debug("[Cache] MISS: %s", cache_key)
                result = func(*args, **kwargs)

                # Extraire la réponse et le statut
                if isinstance(result, tuple):
                    response, status = result
                else:
                    response, status = result, HTTP_OK

                # Mettre en cache seulement les réponses réussies
                if status == HTTP_OK:
                    try:
                        cached_data = {
                            "response": response,
                            "status": status,
                        }
                        redis_client.setex(
                            cache_key,
                            ttl,
                            json.dumps(cached_data, default=str),
                        )
                        logger.debug("[Cache] SET: %s (TTL: %ss)", cache_key, ttl)
                    except (TimeoutError, Exception) as e:
                        # ✅ Timeout Redis : erreur transitoire, ne pas alerter excessivement
                        error_type = (
                            "Timeout" if isinstance(e, TimeoutError) else "Erreur"
                        )
                        logger.debug(
                            "[Cache] %s lors de la mise en cache (non critique): %s",
                            error_type,
                            e,
                        )

                return response, status

            except Exception as e:
                logger.exception("[Cache] Erreur inattendue: %s", e)
                # En cas d'erreur, exécuter normalement
                return func(*args, **kwargs)

        return wrapper

    return decorator


def invalidate_cache(prefix: str, *args: Any, **kwargs: Any) -> bool:
    """
    Invalide une entrée de cache spécifique.

    Args:
        prefix: Préfixe de la clé
        *args: Arguments positionnels pour générer la clé
        **kwargs: Arguments nommés pour générer la clé

    Returns:
        True si la clé a été supprimée, False sinon
    """
    redis_client = _get_redis_client()
    if not redis_client:
        return False

    cache_key = _generate_cache_key(prefix, *args, **kwargs)

    try:
        deleted = redis_client.delete(cache_key)
        if deleted:
            logger.debug("[Cache] INVALIDATED: %s", cache_key)
        return bool(deleted)
    except Exception as e:
        logger.warning("[Cache] Erreur lors de l'invalidation: %s", e)
        return False


def invalidate_cache_pattern(pattern: str) -> int:
    """
    Invalide toutes les clés de cache correspondant au pattern.

    Args:
        pattern: Pattern Redis (ex: "cache:api:rides:*")

    Returns:
        Nombre de clés supprimées
    """
    redis_client = _get_redis_client()
    if not redis_client:
        return 0

    try:
        keys = redis_client.keys(pattern)
        if keys:
            deleted = redis_client.delete(*keys)
            logger.debug("[Cache] INVALIDATED %s keys matching: %s", deleted, pattern)
            return deleted
        return 0
    except Exception as e:
        logger.warning("[Cache] Erreur lors de l'invalidation par pattern: %s", e)
        return 0
