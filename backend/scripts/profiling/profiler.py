"""✅ B3: Système de profiling par fonction avec budgets de performance.

Objectif: Identifier précisément les hotspots et alerter sur dépassements.
"""

from __future__ import annotations

import cProfile
import logging
import os
import pstats
import time
from contextlib import contextmanager
from io import StringIO
from typing import Any, Dict, List

from backend.services.unified_dispatch.performance_metrics import DispatchPerformanceMetrics

logger = logging.getLogger(__name__)

# ✅ B3: Budgets de performance par étape (ms)
PERFORMANCE_BUDGETS = {
    "data_collection": 10000,  # 10s
    "heuristics": 15000,  # 15s
    "solver": 30000,  # 30s
    "persistence": 5000,  # 5s
    "total": 60000,  # 60s
}


class DispatchProfiler:
    """Profiler pour tracking des performances et hotspots."""

    def __init__(self, enabled: bool = False):
        """Initialise le profiler.

        Args:
            enabled: Active le profiling si True
        """
        self.enabled = enabled
        self.profiler = cProfile.Profile() if enabled else None
        self.function_times: Dict[str, float] = {}
        self.stage_start_times: Dict[str, float] = {}

    def is_enabled(self) -> bool:
        """Vérifie si le profiling est activé."""
        return self.enabled and self.profiler is not None

    def start(self) -> None:
        """Démarre le profiler."""
        if self.profiler:
            self.profiler.enable()

    def stop(self) -> None:
        """Arrête le profiler."""
        if self.profiler:
            self.profiler.disable()

    @contextmanager
    def profile_stage(self, stage_name: str):
        """Context manager pour profiler une étape.

        Args:
            stage_name: Nom de l'étape (data_collection, heuristics, etc.)
        """
        start_time = time.time()
        self.stage_start_times[stage_name] = start_time

        try:
            yield
        finally:
            elapsed = (time.time() - start_time) * 1000  # ms
            self.function_times[stage_name] = elapsed
            logger.debug("[Profiling] Stage '%s' took %.2fms", stage_name, elapsed)

    def get_top_functions(self, n: int = 10) -> List[Dict[str, Any]]:
        """Retourne le top N des fonctions les plus lentes.

        Args:
            n: Nombre de fonctions à retourner

        Returns:
            Liste de dict avec 'name', 'time', 'calls', 'cumtime'
        """
        if not self.profiler:
            return []

        s = StringIO()
        stats = pstats.Stats(self.profiler, stream=s)
        stats.sort_stats("cumulative")
        stats.print_stats(n)

        top_functions = []
        lines = s.getvalue().split("\n")[5:-3]  # Skip header/footer

        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    ncalls = parts[0]
                    tottime = float(parts[1])
                    percall = parts[2]
                    cumtime = float(parts[3])
                    name = " ".join(parts[5:])

                    top_functions.append(
                        {
                            "name": name,
                            "ncalls": ncalls,
                            "tottime": tottime,
                            "cumtime": cumtime,
                            "time_per_call": percall,
                        }
                    )
                except (ValueError, IndexError):
                    continue

        return top_functions[:n]

    def check_budgets(self, metrics: DispatchPerformanceMetrics) -> Dict[str, Any]:
        """Vérifie si les budgets de performance sont respectés.

        Args:
            metrics: Métriques de performance du dispatch

        Returns:
            Dict avec budgets checkés et alertes si dépassements
        """
        issues = []
        status = {}

        # Convertir temps en ms
        data_time_ms = metrics.data_collection_time * 1000
        heuristics_time_ms = metrics.heuristics_time * 1000
        solver_time_ms = metrics.solver_time * 1000
        persistence_time_ms = metrics.persistence_time * 1000
        total_time_ms = metrics.total_time * 1000

        # Vérifier chaque budget
        checks = [
            ("data_collection", data_time_ms, PERFORMANCE_BUDGETS["data_collection"]),
            ("heuristics", heuristics_time_ms, PERFORMANCE_BUDGETS["heuristics"]),
            ("solver", solver_time_ms, PERFORMANCE_BUDGETS["solver"]),
            ("persistence", persistence_time_ms, PERFORMANCE_BUDGETS["persistence"]),
            ("total", total_time_ms, PERFORMANCE_BUDGETS["total"]),
        ]

        for stage, actual_ms, budget_ms in checks:
            exceeded = actual_ms > budget_ms
            status[stage] = {
                "actual_ms": round(actual_ms, 2),
                "budget_ms": budget_ms,
                "exceeded": exceeded,
                "pct_of_budget": round((actual_ms / budget_ms) * 100, 1),
            }

            if exceeded:
                issues.append(
                    {
                        "stage": stage,
                        "actual_ms": actual_ms,
                        "budget_ms": budget_ms,
                        "over_budget": actual_ms - budget_ms,
                    }
                )

        return {"all_respected": len(issues) == 0, "issues": issues, "budgets": status}

    def generate_report(self, metrics: DispatchPerformanceMetrics | None = None) -> str:
        """Génère un rapport de profiling textuel.

        Args:
            metrics: Métriques optionnelles du dispatch

        Returns:
            Rapport textuel formaté
        """
        lines = []
        lines.append("=" * 80)
        lines.append("B3 PROFILING REPORT")
        lines.append("=" * 80)

        if self.enabled:
            lines.append("\nTOP 10 FUNCTIONS:")
            lines.append("-" * 80)
            top_functions = self.get_top_functions(10)

            for i, func in enumerate(top_functions, 1):
                lines.append(f"{i:2d}. {func['name'][:50]:50s} {func['cumtime']:8.3f}s")
        else:
            lines.append("\nProfiling disabled (set ENABLE_PROFILING=1 to enable)")

        if metrics:
            lines.append("\n\nPERFORMANCE BUDGETS:")
            lines.append("-" * 80)

            budget_check = self.check_budgets(metrics)

            for stage, info in budget_check["budgets"].items():
                status = "⚠️ EXCEEDED" if info["exceeded"] else "✅ OK"
                lines.append(
                    f"{stage:20s}: {info['actual_ms']:8.0f}ms / {info['budget_ms']:8.0f}ms "
                    f"({info['pct_of_budget']:5.1f}%) {status}"
                )

            if budget_check["issues"]:
                lines.append("\n⚠️ ALERTS:")
                for issue in budget_check["issues"]:
                    lines.append(
                        f"  {issue['stage']}: {issue['actual_ms']:.0f}ms > {issue['budget_ms']:.0f}ms "
                        f"(+{issue['over_budget']:.0f}ms)"
                    )

        lines.append("=" * 80)
        return "\n".join(lines)


def is_profiling_enabled() -> bool:
    """Vérifie si le profiling est activé via variable d'environnement."""
    return os.getenv("ENABLE_PROFILING", "0") in ("1", "true", "True")


def get_profiler() -> DispatchProfiler:
    """Retourne une instance du profiler selon la configuration."""
    return DispatchProfiler(enabled=is_profiling_enabled())
