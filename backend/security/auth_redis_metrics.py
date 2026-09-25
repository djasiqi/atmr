"""Collecte des gauges Redis auth (P0-6) — à appeler depuis /metrics ou un job périodique."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def refresh_redis_auth_gauges() -> None:
    """Met à jour used/max/utilization/evicted depuis INFO memory + stats."""
    try:
        from security.auth_redis import get_auth_redis
        from security.security_metrics import (
            redis_auth_evicted_keys_total,
            redis_auth_memory_max_bytes,
            redis_auth_memory_used_bytes,
            redis_auth_memory_utilization,
        )

        client = get_auth_redis()
        info = client.info(section="memory")
        stats = client.info(section="stats")
        used = int(info.get("used_memory") or 0)
        maxmem = int(info.get("maxmemory") or 0)
        evicted = int(stats.get("evicted_keys") or 0)
        redis_auth_memory_used_bytes.set(used)
        redis_auth_memory_max_bytes.set(maxmem)
        util = (float(used) / float(maxmem)) if maxmem > 0 else 0.0
        redis_auth_memory_utilization.set(util)
        redis_auth_evicted_keys_total.set(evicted)
        if evicted > 0:
            logger.critical(
                "auth_redis_evicted_keys_nonzero count=%s — invariant P0-6 violé",
                evicted,
            )
    except Exception:
        logger.debug("refresh_redis_auth_gauges skipped", exc_info=True)
