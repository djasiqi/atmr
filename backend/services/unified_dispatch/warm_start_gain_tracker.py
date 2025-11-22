"""✅ B5: Tracking du gain warm-start OR-Tools.

Objectif: Mesurer le gain de temps avec warm-start vs sans.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ✅ B5: Constantes pour seuils
MIN_GAIN_THRESHOLD_PCT = 10.0  # Alerte si gain < 10%
TARGET_SIZE_MIN = 100  # Taille min pour mesure
TARGET_SIZE_MAX = 200  # Taille max pour mesure
HISTORY_WINDOW_DAYS = 7  # Fenêtre historique (jours)


@dataclass
class WarmStartTiming:
    """Timing d'une résolution avec/sans warm-start."""

    size: int  # Nombre de bookings
    num_vehicles: int
    with_warm_start: bool
    solve_time_ms: float
    solution_found: bool = True


class WarmStartGainTracker:
    """Tracker pour gains warm-start historiques."""

    def __init__(self) -> None:  # type: ignore[override]
        """Initialise le tracker."""
        self.gains_history: List[Dict[str, Any]] = []
        self.median_gain_pct: float = 0.0
        self.avg_gain_pct: float = 0.0
        self.should_alert: bool = False  # True si gain < 10% sur 7j

    def record_comparison(
        self, size: int, with_time: float, without_time: float
    ) -> None:
        """Enregistre une comparaison avec/sans warm-start."""

        if without_time == 0:
            logger.warning("[B5] without_time = 0, skipping gain calculation")
            return

        gain_pct = ((without_time - with_time) / without_time) * 100
        improvement = without_time - with_time

        result = {
            "date": datetime.now(UTC),
            "size": size,
            "without_ms": without_time,
            "with_ms": with_time,
            "gain_pct": gain_pct,
            "improvement_ms": improvement,
        }

        self.gains_history.append(result)

        # Recalculer médian et avg sur les 7 derniers jours (max 100 entrées)
        recent = self.gains_history[-100:]

        if recent:
            gains_pct = [r["gain_pct"] for r in recent]
            gains_pct_sorted = sorted(gains_pct)
            n = len(gains_pct_sorted)

            # Médian
            self.median_gain_pct = (
                gains_pct_sorted[n // 2]
                if n % 2 == 1
                else (gains_pct_sorted[n // 2 - 1] + gains_pct_sorted[n // 2]) / 2
            )

            # Moyenne
            self.avg_gain_pct = sum(gains_pct) / n if n > 0 else 0.0

            # Alerter si gain < 10%
            self.should_alert = self.median_gain_pct < MIN_GAIN_THRESHOLD_PCT

            logger.info(
                "[B5] Warm-start: median=%.1f%%, avg=%.1f%%, should_alert=%s (n=%d)",
                self.median_gain_pct,
                self.avg_gain_pct,
                self.should_alert,
                n,
            )

            if self.should_alert:
                logger.warning(
                    "[B5] ⚠️ Warm-start gain trop faible: %.1f%% < %.1f%% (sur 7j)",
                    self.median_gain_pct,
                    MIN_GAIN_THRESHOLD_PCT,
                )


# Singleton global
_global_gain_tracker: WarmStartGainTracker | None = None


def get_gain_tracker() -> WarmStartGainTracker:
    """Retourne l'instance singleton du tracker."""
    global _global_gain_tracker  # noqa: PLW0603

    if _global_gain_tracker is None:
        _global_gain_tracker = WarmStartGainTracker()

    return _global_gain_tracker


def reset_gain_tracker() -> None:
    """Reset le tracker (pour tests)."""
    global _global_gain_tracker  # noqa: PLW0603
    _global_gain_tracker = WarmStartGainTracker()


def measure_warm_start_gain(
    problem: Dict[str, Any], heuristic_assignments: List[Any], solve_func: Any
) -> Dict[str, Any]:
    """✅ B5: Mesure le gain du warm-start (A/A interne).

    Args:
        problem: Dict du problème VRPTW
        heuristic_assignments: Assignments heuristiques
        solve_func: Fonction solve() à appeler

    Returns:
        Dict avec 'gain_pct', 'with_time', 'without_time', 'size'
    """
    from services.unified_dispatch.settings import Settings

    tracker = get_gain_tracker()

    size = len(problem.get("bookings", []))
    num_vehicles = problem.get("num_vehicles", 0)

    # Taille cible: 100-200 bookings
    if not (TARGET_SIZE_MIN <= size <= TARGET_SIZE_MAX):
        logger.debug(
            "[B5] Skipping gain measurement (size=%d not in %d-%d)",
            size,
            TARGET_SIZE_MIN,
            TARGET_SIZE_MAX,
        )
        return {
            "skipped": True,
            "reason": f"size={size} not in {TARGET_SIZE_MIN}-{TARGET_SIZE_MAX}",
        }

    try:
        # Test 1: SANS warm-start
        problem_without = problem.copy()
        problem_without.pop("heuristic_assignments", None)

        start_time = time.time()
        _ = solve_func(problem_without, Settings())
        without_time_ms = (time.time() - start_time) * 1000

        # Test 2: AVEC warm-start
        problem_with = problem.copy()
        problem_with["heuristic_assignments"] = heuristic_assignments

        start_time = time.time()
        _ = solve_func(problem_with, Settings())
        with_time_ms = (time.time() - start_time) * 1000

        # Calculer gain
        tracker.record_comparison(size, with_time_ms, without_time_ms)

        return {
            "skipped": False,
            "size": size,
            "num_vehicles": num_vehicles,
            "without_ms": without_time_ms,
            "with_ms": with_time_ms,
            "gain_pct": tracker.median_gain_pct,
            "should_alert": tracker.should_alert,
        }

    except Exception as e:
        logger.warning("[B5] Failed to measure warm-start gain: %s", e)
        return {"skipped": True, "reason": str(e)}
