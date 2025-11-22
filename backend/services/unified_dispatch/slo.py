"""✅ A4: Définition des SLO (Service Level Objectives) pour le dispatch.

SLO déclarés:
- Latence dispatch par taille de batch
- Taux de réussite (assignment rate)
- Quality score minimum
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

logger = __import__("logging").getLogger(__name__)

# Constants pour seuils de taille de batch
SMALL_BATCH_THRESHOLD = 50
LARGE_BATCH_THRESHOLD = 200

# Constants pour severity thresholds
WARNING_BREACH_COUNT = 3
CRITICAL_BREACH_COUNT = 5

# ==================== SLO Definitions ====================


@dataclass
class SLOTarget:
    """Objectif SLO avec seuils."""

    latency_p50_max_ms: int  # 50e percentile latence max
    latency_p95_max_ms: int  # 95e percentile latence max
    latency_p99_max_ms: int  # 99e percentile latence max
    success_rate_min: float  # Taux de réussite min (0-1)
    quality_score_min: float  # Quality score min (0-100)
    breach_window_minutes: int = 15  # Fenêtre pour détecter breach
    breach_threshold_count: int = 3  # Nombre de breaches pour alerter


# ✅ A4: SLO par taille de batch (taille = nombre de bookings)
SLO_BY_BATCH_SIZE = {
    # Petits batches (< 50 bookings)
    "small": SLOTarget(
        latency_p50_max_ms=5000,  # 5s
        latency_p95_max_ms=10000,  # 10s
        latency_p99_max_ms=15000,  # 15s
        success_rate_min=0.95,  # 95%
        quality_score_min=80.0,
    ),
    # Batches moyens (50-200 bookings)
    "medium": SLOTarget(
        latency_p50_max_ms=15000,  # 15s
        latency_p95_max_ms=30000,  # 30s
        latency_p99_max_ms=45000,  # 45s
        success_rate_min=0.90,  # 90%
        quality_score_min=75.0,
    ),
    # Grands batches (> 200 bookings)
    "large": SLOTarget(
        latency_p50_max_ms=30000,  # 30s
        latency_p95_max_ms=60000,  # 60s
        latency_p99_max_ms=90000,  # 90s
        success_rate_min=0.85,  # 85%
        quality_score_min=70.0,
    ),
}


def get_slo_for_batch_size(n_bookings: int) -> SLOTarget:
    """Retourne le SLO cible pour une taille de batch donnée.

    Args:
        n_bookings: Nombre de bookings à traiter

    Returns:
        SLOTarget correspondant à la taille du batch
    """
    if n_bookings < SMALL_BATCH_THRESHOLD:
        return SLO_BY_BATCH_SIZE["small"]
    if n_bookings < LARGE_BATCH_THRESHOLD:
        return SLO_BY_BATCH_SIZE["medium"]
    return SLO_BY_BATCH_SIZE["large"]


def check_slo_breach(
    total_time_sec: float, assignment_rate: float, quality_score: float, n_bookings: int
) -> dict[str, Any]:
    """Vérifie si les métriques dépassent les SLO.

    Args:
        total_time_sec: Temps total d'exécution (secondes)
        assignment_rate: Taux d'assignation (0-1)
        quality_score: Score de qualité (0-100)
        n_bookings: Nombre de bookings

    Returns:
        Dict avec:
        - breached: bool
        - slo_target: SLOTarget
        - breaches: liste des violations détectées
        - latency_breach: bool
        - success_breach: bool
        - quality_breach: bool
    """
    slo = get_slo_for_batch_size(n_bookings)

    # Convertir latence en ms pour comparaison
    total_time_ms = total_time_sec * 1000

    # Vérifier chaque dimension
    latency_breach = total_time_ms > slo.latency_p95_max_ms
    success_breach = assignment_rate < slo.success_rate_min
    quality_breach = quality_score < slo.quality_score_min

    breaches = []
    if latency_breach:
        breaches.append(
            {
                "dimension": "latency",
                "actual": f"{total_time_ms:.0f}ms",
                "threshold": f"{slo.latency_p95_max_ms}ms",
                "severity": "warning",
            }
        )
    if success_breach:
        breaches.append(
            {
                "dimension": "success_rate",
                "actual": f"{assignment_rate:.2%}",
                "threshold": f"{slo.success_rate_min:.2%}",
                "severity": "critical",
            }
        )
    if quality_breach:
        breaches.append(
            {
                "dimension": "quality_score",
                "actual": f"{quality_score:.1f}",
                "threshold": f"{slo.quality_score_min:.1f}",
                "severity": "warning",
            }
        )

    return {
        "breached": len(breaches) > 0,
        "slo_target": {
            "category": "small"
            if n_bookings < SMALL_BATCH_THRESHOLD
            else ("medium" if n_bookings < LARGE_BATCH_THRESHOLD else "large"),
            "n_bookings": n_bookings,
            "latency_p95_max_ms": slo.latency_p95_max_ms,
            "success_rate_min": slo.success_rate_min,
            "quality_score_min": slo.quality_score_min,
        },
        "breaches": breaches,
        "latency_breach": latency_breach,
        "success_breach": success_breach,
        "quality_breach": quality_breach,
    }


# ==================== Historical SLO Tracking ====================


class SLOBreachTracker:
    """Track les breaches SLO dans une fenêtre glissante pour alertes."""

    def __init__(self, window_minutes: int = 15, breach_threshold: int = 3):
        super().__init__()
        """Initialise le tracker.

        Args:
            window_minutes: Fenêtre temporelle (minutes)
            breach_threshold: Seuil de breaches pour alerter
        """
        self.window_minutes = window_minutes
        self.breach_threshold = breach_threshold
        self._breaches: list[Dict[str, Any]] = []

    def record_breach(self, breach_type: str, timestamp: float) -> None:
        """Enregistre une breach.

        Args:
            breach_type: Type de breach ('latency', 'success', 'quality')
            timestamp: Timestamp Unix
        """
        self._breaches.append(
            {
                "type": breach_type,
                "timestamp": timestamp,
            }
        )

    def get_recent_breaches(self, current_time: float) -> list[Dict[str, Any]]:
        """Retourne les breaches dans la fenêtre temporelle.

        Args:
            current_time: Timestamp Unix courant

        Returns:
            Liste des breaches dans la fenêtre
        """
        cutoff = current_time - (self.window_minutes * 60)
        return [b for b in self._breaches if b["timestamp"] >= cutoff]

    def should_alert(self, current_time: float) -> bool:
        """Vérifie si on doit alerter (>= seuil breaches dans fenêtre).

        Args:
            current_time: Timestamp Unix courant

        Returns:
            True si on doit alerter
        """
        recent = self.get_recent_breaches(current_time)
        return len(recent) >= self.breach_threshold

    def get_breach_summary(self, current_time: float) -> Dict[str, Any]:
        """Retourne un résumé des breaches dans la fenêtre.

        Args:
            current_time: Timestamp Unix courant

        Returns:
            Dict avec count, by_type, severity
        """
        recent = self.get_recent_breaches(current_time)

        by_type: dict[str, int] = {}
        for breach in recent:
            btype = breach["type"]
            by_type[btype] = by_type.get(btype, 0) + 1

        severity = (
            "critical"
            if len(recent) >= CRITICAL_BREACH_COUNT
            else ("warning" if len(recent) >= WARNING_BREACH_COUNT else "info")
        )

        return {
            "breach_count": len(recent),
            "by_type": by_type,
            "severity": severity,
            "window_minutes": self.window_minutes,
            "threshold": self.breach_threshold,
            "should_alert": len(recent) >= self.breach_threshold,
        }


# Instance globale pour tracking SLO
_global_slo_tracker = SLOBreachTracker()


def get_slo_tracker() -> SLOBreachTracker:
    """Retourne l'instance globale du tracker SLO."""
    return _global_slo_tracker


def reset_slo_tracker() -> None:
    """Réinitialise le tracker SLO (pour tests)."""
    # module-level mutable is acceptable for singleton
    # Réinitialiser l'instance existante plutôt que d'appeler __init__ directement
    _global_slo_tracker.window_minutes = 15
    _global_slo_tracker.breach_threshold = 3
    _global_slo_tracker.breaches.clear()
