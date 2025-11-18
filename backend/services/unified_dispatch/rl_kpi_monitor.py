# backend/services/unified_dispatch/rl_kpi_monitor.py
"""Monitor pour KPIs RL avec backout automatique."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

try:
    from ext import redis_client
except ImportError:
    redis_client = None  # Type: ignore
    logging.warning("[RLKPIMonitor] Redis client not available")

logger = logging.getLogger(__name__)


class RLKPIMonitor:
    """Monitor les KPIs RL et déclenche backout si nécessaire."""

    def __init__(self, settings: Any):
        """Initialise le monitor KPI.

        Args:
            settings: Configuration settings avec RLSettings
        """
        super().__init__()
        self.settings = settings
        self.failure_count_key = "rl:failures:{company_id}"
        self.kpi_history_key = "rl:kpi_history:{company_id}"

    def check_kpis(self, company_id: int, kpis: Dict[str, Any]) -> Tuple[bool, str]:
        """Vérifie les KPIs et détermine si backout nécessaire.

        Args:
            company_id: ID de l'entreprise
            kpis: Dict des KPIs actuels

        Returns:
            Tuple (should_backout, reason)
        """
        # Vérifier seuils
        violations = []

        quality_score = float(kpis.get("quality_score", 0))
        on_time_rate = float(kpis.get("on_time_rate", 0))
        avg_delay_min = float(kpis.get("avg_delay_min", float("inf")))

        # Vérifier quality_score
        min_quality_score = getattr(self.settings.rl, "min_quality_score", 70.0)
        if quality_score < min_quality_score:
            violations.append(f"quality_score={quality_score:.1f} < {min_quality_score}")

        # Vérifier on_time_rate
        min_on_time_rate = getattr(self.settings.rl, "min_on_time_rate", 85.0)
        if on_time_rate < min_on_time_rate:
            violations.append(f"on_time_rate={on_time_rate:.1f}% < {min_on_time_rate}%")

        # Vérifier avg_delay
        max_avg_delay = getattr(self.settings.rl, "max_avg_delay_min", 5.0)
        if avg_delay_min > max_avg_delay:
            violations.append(f"avg_delay={avg_delay_min:.1f}min > {max_avg_delay}min")

        if violations:
            logger.warning("[RLKPIMonitor] Company %d: KPI violations: %s", company_id, ", ".join(violations))
            self._increment_failures(company_id)

            # Vérifier si backout nécessaire
            if self._should_backout(company_id):
                reason = f"Consecutive failures: {', '.join(violations)}"
                logger.error("[RLKPIMonitor] Company %d: BACKOUT TRIGGERED - %s", company_id, reason)
                return True, reason
        else:
            # Reset failure count on success
            self._reset_failures(company_id)
            logger.debug("[RLKPIMonitor] Company %d: All KPIs OK, failure count reset", company_id)

        return False, ""

    def get_failure_count(self, company_id: int) -> int:
        """Retourne le nombre de failures consécutives.

        Args:
            company_id: ID de l'entreprise

        Returns:
            Nombre de failures
        """
        if not redis_client:
            return 0

        key = self.failure_count_key.format(company_id=company_id)
        failures = redis_client.get(key)
        # Type: ignore car redis_client.get() retourne un type générique
        return int(failures) if failures else 0  # type: ignore[arg-type]

    def _increment_failures(self, company_id: int) -> None:
        """Incrémente le compteur de failures."""
        if not redis_client:
            return

        key = self.failure_count_key.format(company_id=company_id)
        try:
            redis_client.incr(key)
            redis_client.expire(key, 86400)  # TTL 24h
        except Exception as e:
            logger.error("[RLKPIMonitor] Failed to increment failures for company %d: %s", company_id, e)

    def _should_backout(self, company_id: int) -> bool:
        """Vérifie si backout nécessaire."""
        if not redis_client:
            return False

        key = self.failure_count_key.format(company_id=company_id)
        try:
            failures = redis_client.get(key)
            # Type: ignore car redis_client.get() retourne un type générique
            failure_count = int(failures) if failures else 0  # type: ignore[arg-type]

            threshold = getattr(self.settings.rl, "consecutive_failures_threshold", 2)

            return failure_count >= threshold
        except Exception as e:
            logger.error("[RLKPIMonitor] Failed to check backout for company %d: %s", company_id, e)
            return False

    def _reset_failures(self, company_id: int) -> None:
        """Reset le compteur de failures."""
        if not redis_client:
            return

        key = self.failure_count_key.format(company_id=company_id)
        try:
            redis_client.delete(key)
        except Exception as e:
            logger.error("[RLKPIMonitor] Failed to reset failures for company %d: %s", company_id, e)
