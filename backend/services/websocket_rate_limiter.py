# backend/services/websocket_rate_limiter.py

"""Rate limiter pour events Socket.IO individuels.

Étend le rate limiting existant (connexions) pour couvrir tous les events.
Utilise Redis sliding window si disponible, sinon cache mémoire.
"""

import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from ext import redis_client

logger = logging.getLogger(__name__)

# Limites par event (event_name -> (limit, window_seconds))
RATE_LIMITS = {
    "connect": (
        20,
        60,
    ),  # ✅ 20 req/min par IP (augmenté de 5 à 20 pour supporter NAT/multi-drivers)
    "driver_location": (1, 1),  # 1 req/s par driver
    "driver_location_batch": (1, 5),  # 1 req/5s par driver
    "team_chat_message": (10, 60),  # 10 req/min par user
    "typing_start": (20, 60),  # 20 req/min par user
    "typing_stop": (20, 60),  # 20 req/min par user
}

# Cache mémoire fallback (si Redis indisponible)
_memory_cache: dict[str, list[datetime]] = defaultdict(list)


class WebSocketRateLimiter:
    """Rate limiter par event et par utilisateur/driver/IP."""

    def __init__(self):  # pyright: ignore[reportMissingSuperCall]
        """Initialise le rate limiter.

        Note: Pas de classe parente, donc pas besoin d'appeler super().__init__()
        """
        self.use_redis = redis_client is not None

    def check_rate_limit(
        self,
        event: str,
        user_id: int | None = None,
        driver_id: int | None = None,
        client_ip: str | None = None,
    ) -> tuple[bool, int | None]:
        """Vérifie si un event respecte le rate limit.

        Args:
            event: Nom de l'event Socket.IO
            user_id: ID de l'utilisateur (pour events user-scoped)
            driver_id: ID du chauffeur (pour events driver-scoped)
            client_ip: IP du client (pour events IP-scoped)

        Returns:
            Tuple (allowed, retry_after_seconds):
            - allowed: True si rate limit OK, False sinon
            - retry_after_seconds: Nombre de secondes à attendre (None si allowed=True)
        """
        # Vérifier si l'event a une limite définie
        if event not in RATE_LIMITS:
            # Pas de limite pour cet event
            return True, None

        limit, window_seconds = RATE_LIMITS[event]

        # Déterminer la clé de rate limiting selon le type d'event
        if event == "connect":
            # Rate limit par IP pour les connexions
            if not client_ip:
                logger.warning("Rate limit connect sans client_ip")
                return True, None
            key = f"ws_rate_limit:{event}:ip:{client_ip}"
            identifier = client_ip
        elif event in ("driver_location", "driver_location_batch"):
            # Rate limit par driver pour les positions
            if not driver_id:
                logger.warning("Rate limit %s sans driver_id", event)
                return True, None
            key = f"ws_rate_limit:{event}:driver:{driver_id}"
            identifier = f"driver_{driver_id}"
        else:
            # Rate limit par user pour les autres events
            if not user_id:
                logger.warning("Rate limit %s sans user_id", event)
                return True, None
            key = f"ws_rate_limit:{event}:user:{user_id}"
            identifier = f"user_{user_id}"

        # Utiliser Redis si disponible
        if self.use_redis:
            return self._check_redis(key, limit, window_seconds)

        # Fallback cache mémoire
        return self._check_memory(key, identifier, limit, window_seconds)

    def _check_redis(
        self, key: str, limit: int, window_seconds: int
    ) -> tuple[bool, int | None]:
        """Vérifie le rate limit avec Redis (sliding window).

        Args:
            key: Clé Redis
            limit: Limite d'events
            window_seconds: Fenêtre en secondes

        Returns:
            Tuple (allowed, retry_after_seconds)
        """
        if not redis_client:
            # Redis non disponible, fallback mémoire
            return True, None

        try:
            now = datetime.now(UTC)
            now_ts = int(now.timestamp())

            # Sliding window: garder seulement les timestamps dans la fenêtre
            window_start_ts = now_ts - window_seconds

            # Utiliser une liste Redis pour stocker les timestamps
            pipe = redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start_ts)  # Nettoyer anciens
            pipe.zcard(key)  # Compter events dans la fenêtre
            pipe.zadd(key, {str(now_ts): now_ts})  # Ajouter timestamp actuel
            pipe.expire(
                key, window_seconds + 1
            )  # TTL légèrement supérieur à la fenêtre
            results = pipe.execute()

            count = (
                results[1] if results and len(results) > 1 else 0
            )  # Nombre d'events dans la fenêtre (avant ajout)

            if count >= limit:
                # Rate limit dépassé
                # Calculer le temps à attendre (premier timestamp dans la fenêtre)
                oldest_ts_list = redis_client.zrange(key, 0, 0, withscores=True)
                if (
                    oldest_ts_list
                    and isinstance(oldest_ts_list, list)
                    and len(oldest_ts_list) > 0
                ):
                    oldest_tuple = oldest_ts_list[0]
                    if (
                        isinstance(oldest_tuple, (tuple, list))
                        and len(oldest_tuple) > 1
                    ):
                        oldest = int(oldest_tuple[1])
                        retry_after = (oldest + window_seconds) - now_ts
                        retry_after = max(1, retry_after)  # Au moins 1 seconde
                    else:
                        retry_after = window_seconds
                else:
                    retry_after = window_seconds

                logger.warning(
                    "Rate limit dépassé: event=%s, key=%s, count=%d, limit=%d, retry_after=%d",
                    key.split(":")[1],
                    key,
                    count + 1,
                    limit,
                    retry_after,
                )
                return False, retry_after

            # Rate limit OK
            return True, None

        except Exception as e:
            logger.warning("Erreur Redis rate limiting (fail-open): %s", e)
            # Fail-open: autoriser l'event en cas d'erreur Redis
            return True, None

    def _check_memory(
        self, key: str, identifier: str, limit: int, window_seconds: int
    ) -> tuple[bool, int | None]:
        """Vérifie le rate limit avec cache mémoire (fallback).

        Args:
            key: Clé (pour logging)
            identifier: Identifiant (pour cache mémoire)
            limit: Limite d'events
            window_seconds: Fenêtre en secondes

        Returns:
            Tuple (allowed, retry_after_seconds)
        """
        now = datetime.now(UTC)
        window_start = now - timedelta(seconds=window_seconds)

        # Nettoyer anciens timestamps
        cache_key = f"{key}:{identifier}"
        _memory_cache[cache_key] = [
            ts for ts in _memory_cache[cache_key] if ts > window_start
        ]

        # Compter events dans la fenêtre
        count = len(_memory_cache[cache_key])

        if count >= limit:
            # Rate limit dépassé
            if _memory_cache[cache_key]:
                oldest = _memory_cache[cache_key][0]
                retry_after = int(
                    (oldest + timedelta(seconds=window_seconds) - now).total_seconds()
                )
                retry_after = max(1, retry_after)
            else:
                retry_after = window_seconds

            logger.warning(
                "Rate limit dépassé (mémoire): event=%s, key=%s, count=%d, limit=%d, retry_after=%d",
                key.split(":")[1],
                key,
                count + 1,
                limit,
                retry_after,
            )
            return False, retry_after

        # Ajouter timestamp actuel
        _memory_cache[cache_key].append(now)

        # Rate limit OK
        return True, None


# Instance globale
ws_rate_limiter = WebSocketRateLimiter()
