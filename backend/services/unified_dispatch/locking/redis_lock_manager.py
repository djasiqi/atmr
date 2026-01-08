"""Gestionnaire de verrous distribués Redis pour éviter les runs concurrents."""

import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


class RedisLockManager:
    """Gestionnaire de verrous distribués Redis pour le dispatch.

    Utilise des verrous Redis avec TTL pour éviter les runs concurrents
    sur la même entreprise et la même date.
    """

    LOCK_TTL_SECONDS = 300  # 5 minutes
    LOCK_KEY_PREFIX = "dispatch:lock"

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        """Initialise le gestionnaire de verrous."""
        self._redis_client: Any = None

    def _get_redis_client(self) -> Any:
        """Récupère le client Redis (lazy loading)."""
        if self._redis_client is None:
            from ext import redis_client

            self._redis_client = redis_client
        return self._redis_client

    def acquire_lock(
        self, lock_key: str, *, timeout_seconds: int | None = None
    ) -> bool:
        """Acquiert un verrou Redis via une clé explicite (API compatible orchestrator).

        Args:
            lock_key: Clé logique du verrou (ex: "dispatch:42:2025-01-01").
            timeout_seconds: TTL du verrou (défaut: `LOCK_TTL_SECONDS`).

        Returns:
            True si acquis, False sinon.
        """
        redis_client = self._get_redis_client()
        if not redis_client:
            return False

        ttl = (
            int(timeout_seconds)
            if timeout_seconds is not None
            else self.LOCK_TTL_SECONDS
        )
        # Normaliser: on préfixe toutes les clés pour éviter collisions
        key = (
            lock_key
            if lock_key.startswith(self.LOCK_KEY_PREFIX + ":")
            else f"{self.LOCK_KEY_PREFIX}:{lock_key}"
        )
        try:
            result = redis_client.set(key, "1", nx=True, ex=ttl)
            return result is True
        except Exception as e:
            logger.warning(
                "[RedisLockManager] Failed to acquire lock key=%s: %s", key, e
            )
            return False

    def release_lock(self, lock_key: str) -> None:
        """Libère un verrou Redis via une clé explicite
        (API compatible orchestrator)."""
        redis_client = self._get_redis_client()
        if not redis_client:
            return

        key = (
            lock_key
            if lock_key.startswith(self.LOCK_KEY_PREFIX + ":")
            else f"{self.LOCK_KEY_PREFIX}:{lock_key}"
        )
        try:
            redis_client.delete(key)
        except Exception as e:
            logger.warning(
                "[RedisLockManager] Failed to release lock key=%s: %s", key, e
            )

    def acquire(self, company_id: int, day_str: str) -> bool:
        """Acquiert un verrou distribué Redis.

        Args:
            company_id: ID de l'entreprise
            day_str: Date au format YYYY-MM-DD

        Returns:
            True si le verrou a été acquis, False sinon
        """
        # Compat API historique (company_id, day_str)
        return self.acquire_lock(
            f"{company_id}:{day_str}", timeout_seconds=self.LOCK_TTL_SECONDS
        )

    def release(self, company_id: int, day_str: str) -> None:
        """Libère le verrou distribué Redis.

        Args:
            company_id: ID de l'entreprise
            day_str: Date au format YYYY-MM-DD
        """
        # Compat API historique (company_id, day_str)
        self.release_lock(f"{company_id}:{day_str}")

    @contextmanager
    def lock_context(self, company_id: int, day_str: str):
        """Context manager pour acquérir et libérer automatiquement un verrou.

        Args:
            company_id: ID de l'entreprise
            day_str: Date au format YYYY-MM-DD

        Yields:
            True si le verrou a été acquis, False sinon

        Example:
            with lock_manager.lock_context(company_id, day_str) as acquired:
                if acquired:
                    # Code protégé par le verrou
                    pass
        """
        acquired = self.acquire(company_id, day_str)
        try:
            yield acquired
        finally:
            if acquired:
                self.release(company_id, day_str)
