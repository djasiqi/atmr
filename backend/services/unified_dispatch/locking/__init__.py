"""Module de gestion des verrous distribués Redis pour le dispatch."""

from services.unified_dispatch.locking.redis_lock_manager import (
    RedisLockManager,
)

__all__ = ["RedisLockManager"]
