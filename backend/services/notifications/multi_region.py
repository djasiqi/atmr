# backend/services/notifications/multi_region.py
"""Multi-region failover pour notifications ultra-disponibles.

Architecture:
- Primary region (EU-West-1)
- Failover region (EU-Central-1)
- Health checks automatiques
- Basculement transparent en cas de panne
"""

from __future__ import annotations

import logging
import os
import time
from enum import Enum
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Configuration multi-région
PRIMARY_REGION = os.getenv("PRIMARY_REGION", "eu-west-1")
FAILOVER_REGION = os.getenv("FAILOVER_REGION", "eu-central-1")
CURRENT_REGION = os.getenv("CURRENT_REGION", PRIMARY_REGION)

# URLs des services par région
REGION_ENDPOINTS = {
    "eu-west-1": {
        "expo_push": "https://exp.host/--/api/v2/push/send",
        "redis": os.getenv("REDIS_PRIMARY_URL", "redis://redis-primary:6379/0"),
        "celery_broker": os.getenv(
            "CELERY_PRIMARY_URL", "redis://redis-primary:6379/0"
        ),
    },
    "eu-central-1": {
        "expo_push": "https://exp.host/--/api/v2/push/send",  # Expo est global
        "redis": os.getenv("REDIS_FAILOVER_URL", "redis://redis-failover:6379/0"),
        "celery_broker": os.getenv(
            "CELERY_FAILOVER_URL", "redis://redis-failover:6379/0"
        ),
    },
}

# Health check intervals
HEALTH_CHECK_INTERVAL = 30  # secondes
HEALTH_CHECK_TIMEOUT = 5  # secondes
MAX_FAILED_CHECKS = 3  # Nombre d'échecs avant basculement
HIGH_LATENCY_THRESHOLD_MS = 1000  # 1 seconde en millisecondes


class RegionStatus(Enum):
    """Status d'une région."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class RegionManager:
    """Gestionnaire multi-région avec failover automatique."""

    def __init__(self):
        super().__init__()
        self.current_region = CURRENT_REGION
        self.region_status: Dict[str, RegionStatus] = {
            PRIMARY_REGION: RegionStatus.HEALTHY,
            FAILOVER_REGION: RegionStatus.HEALTHY,
        }
        self.failed_checks: Dict[str, int] = {
            PRIMARY_REGION: 0,
            FAILOVER_REGION: 0,
        }
        self.last_check: Dict[str, float] = {
            PRIMARY_REGION: 0,
            FAILOVER_REGION: 0,
        }

    def get_active_region(self) -> str:
        """Retourne la région active actuelle.

        Returns:
            Nom de la région active
        """
        # Vérifier la santé de la région actuelle
        if self._needs_health_check(self.current_region):
            self._check_region_health(self.current_region)

        # Si région actuelle DOWN, basculer
        if self.region_status[self.current_region] == RegionStatus.DOWN:
            logger.warning(
                "[multi_region] Current region %s is DOWN, failing over...",
                self.current_region,
            )
            self._failover()

        return self.current_region

    def get_endpoint(self, service: str) -> str:
        """Retourne l'endpoint d'un service dans la région active.

        Args:
            service: Nom du service (expo_push, redis, celery_broker)

        Returns:
            URL de l'endpoint
        """
        active_region = self.get_active_region()
        endpoints = REGION_ENDPOINTS.get(active_region, {})

        endpoint = endpoints.get(service)
        if not endpoint:
            logger.error(
                "[multi_region] No endpoint for service %s in region %s",
                service,
                active_region,
            )
            # Fallback sur région primaire
            return REGION_ENDPOINTS[PRIMARY_REGION].get(service, "")

        return endpoint

    def _needs_health_check(self, region: str) -> bool:
        """Vérifie si un health check est nécessaire.

        Args:
            region: Nom de la région

        Returns:
            True si health check nécessaire
        """
        last_check = self.last_check.get(region, 0)
        return time.time() - last_check > HEALTH_CHECK_INTERVAL

    def _check_region_health(self, region: str) -> RegionStatus:
        """Vérifie la santé d'une région.

        Args:
            region: Nom de la région

        Returns:
            Status de la région
        """
        try:
            import redis

            logger.debug("[multi_region] Checking health of region %s", region)

            # Vérifier Redis (proxy pour la santé globale)
            redis_url = REGION_ENDPOINTS[region]["redis"]
            r = redis.from_url(redis_url, socket_connect_timeout=HEALTH_CHECK_TIMEOUT)

            # Ping Redis
            start = time.time()
            r.ping()
            latency = (time.time() - start) * 1000  # ms

            logger.debug(
                "[multi_region] Region %s is healthy (latency: %.2fms)",
                region,
                latency,
            )

            # Réinitialiser compteur d'échecs
            self.failed_checks[region] = 0
            self.region_status[region] = RegionStatus.HEALTHY
            self.last_check[region] = time.time()

            # Marquer DEGRADED si latence élevée
            if latency > HIGH_LATENCY_THRESHOLD_MS:
                logger.warning(
                    "[multi_region] Region %s is DEGRADED (high latency: %.2fms)",
                    region,
                    latency,
                )
                self.region_status[region] = RegionStatus.DEGRADED

            return self.region_status[region]

        except Exception as e:
            logger.error(
                "[multi_region] Health check failed for region %s: %s",
                region,
                e,
            )

            # Incrémenter compteur d'échecs
            self.failed_checks[region] += 1
            self.last_check[region] = time.time()

            # Marquer DOWN après MAX_FAILED_CHECKS
            if self.failed_checks[region] >= MAX_FAILED_CHECKS:
                logger.error(
                    "[multi_region] Region %s marked as DOWN after %d failed checks",
                    region,
                    self.failed_checks[region],
                )
                self.region_status[region] = RegionStatus.DOWN
            else:
                self.region_status[region] = RegionStatus.DEGRADED

            return self.region_status[region]

    def _failover(self) -> None:
        """Bascule vers la région de failover."""
        old_region = self.current_region

        # Déterminer la région de failover
        if self.current_region == PRIMARY_REGION:
            new_region = FAILOVER_REGION
        else:
            new_region = PRIMARY_REGION

        # Vérifier la santé de la nouvelle région
        self._check_region_health(new_region)

        if self.region_status[new_region] == RegionStatus.DOWN:
            logger.critical(
                "[multi_region] CRITICAL: Both regions are DOWN! Cannot failover."
            )
            # Garder la région actuelle (mode dégradé)
            return

        # Basculer
        self.current_region = new_region
        logger.warning(
            "[multi_region] ✅ Failover completed: %s -> %s",
            old_region,
            new_region,
        )

        # TODO: Envoyer alerte (PagerDuty / email)
        self._send_failover_alert(old_region, new_region)

    def _send_failover_alert(self, old_region: str, new_region: str) -> None:
        """Envoie une alerte de failover.

        Args:
            old_region: Ancienne région
            new_region: Nouvelle région
        """
        try:
            # TODO: Intégration PagerDuty / email
            logger.critical(
                "[multi_region] 🚨 FAILOVER ALERT: Region switched from %s to %s",
                old_region,
                new_region,
            )
        except Exception as e:
            logger.error("[multi_region] Failed to send failover alert: %s", e)

    def get_status(self) -> Dict[str, Any]:
        """Retourne le status complet multi-région.

        Returns:
            Dict avec status de toutes les régions
        """
        return {
            "current_region": self.current_region,
            "regions": {
                region: {
                    "status": status.value,
                    "failed_checks": self.failed_checks[region],
                    "last_check": self.last_check[region],
                }
                for region, status in self.region_status.items()
            },
        }


# Instance globale du gestionnaire multi-région
region_manager = RegionManager()


def get_active_region() -> str:
    """Retourne la région active actuelle.

    Returns:
        Nom de la région active
    """
    return region_manager.get_active_region()


def get_service_endpoint(service: str) -> str:
    """Retourne l'endpoint d'un service dans la région active.

    Args:
        service: Nom du service (expo_push, redis, celery_broker)

    Returns:
        URL de l'endpoint
    """
    return region_manager.get_endpoint(service)


def get_multi_region_status() -> Dict[str, Any]:
    """Retourne le status complet multi-région.

    Returns:
        Dict avec status de toutes les régions
    """
    return region_manager.get_status()
