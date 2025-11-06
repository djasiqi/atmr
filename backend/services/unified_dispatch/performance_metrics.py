# backend/services/unified_dispatch/performance_metrics.py
"""Métriques de performance pour le dispatch.

Mesure les temps d'exécution par étape (data→heuristics→solver→persist)
et exporte les métriques pour observabilité (Prometheus/StatsD compatible).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class DispatchPerformanceMetrics:
    """Métriques de performance pour un dispatch."""

    # Identifiants
    dispatch_run_id: int | None = None
    company_id: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    # Temps d'exécution par étape (secondes)
    data_collection_time: float = 0.0
    heuristics_time: float = 0.0
    solver_time: float = 0.0
    persistence_time: float = 0.0
    total_time: float = 0.0

    # Compteurs
    sql_queries_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    bookings_processed: int = 0
    drivers_available: int = 0
    temporal_conflicts_count: int = 0  # A1: Conflits temporels détectés
    db_conflicts_count: int = 0  # A2: Conflits DB (violations contraintes uniques)
    osrm_cache_hits: int = 0  # A5: OSRM cache hits
    osrm_cache_misses: int = 0  # A5: OSRM cache misses
    osrm_cache_bypass_count: int = 0  # A5: OSRM cache bypass (Redis HS)

    # Métriques qualité
    quality_score: float = 0.0
    assignment_rate: float = 0.0

    # Métadonnées
    algorithm_used: str = "unknown"
    feature_flags: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour export."""
        return {
            "dispatch_run_id": self.dispatch_run_id,
            "company_id": self.company_id,
            "timestamp": self.timestamp.isoformat(),
            "timing": {
                "data_collection": round(self.data_collection_time, 3),
                "heuristics": round(self.heuristics_time, 3),
                "solver": round(self.solver_time, 3),
                "persistence": round(self.persistence_time, 3),
                "total": round(self.total_time, 3),
            },
            "counters": {
                "sql_queries": self.sql_queries_count,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "bookings_processed": self.bookings_processed,
                "drivers_available": self.drivers_available,
                "temporal_conflicts": self.temporal_conflicts_count,
                "db_conflicts": self.db_conflicts_count,  # A2: Conflits DB
                "osrm_cache_hits": self.osrm_cache_hits,  # A5: OSRM cache
                "osrm_cache_misses": self.osrm_cache_misses,
                "osrm_cache_bypass": self.osrm_cache_bypass_count,
            },
            "quality": {
                "quality_score": self.quality_score,
                "assignment_rate": self.assignment_rate,
            },
            "algorithm": self.algorithm_used,
            "feature_flags": self.feature_flags,
        }

    def to_prometheus_format(self) -> str:
        """Exporte au format Prometheus."""
        lines = [
            "# HELP dispatch_total_time Total time for dispatch (seconds)",
            "# TYPE dispatch_total_time gauge",
            f'dispatch_total_time{{company_id="{self.company_id}"}} {self.total_time:.3f}',
            "",
            "# HELP dispatch_quality_score Quality score (0-100)",
            "# TYPE dispatch_quality_score gauge",
            f'dispatch_quality_score{{company_id="{self.company_id}"}} {self.quality_score:.1f}',
            "",
            "# HELP dispatch_assignment_rate Assignment rate (%)",
            "# TYPE dispatch_assignment_rate gauge",
            f'dispatch_assignment_rate{{company_id="{self.company_id}"}} {self.assignment_rate:.1f}',
            "",
            "# HELP sql_queries_count SQL queries count",
            "# TYPE sql_queries_count counter",
            f'sql_queries_count{{company_id="{self.company_id}"}} {self.sql_queries_count}',
            "",
            "# HELP cache_hit_rate Cache hit rate (0-1)",
            "# TYPE cache_hit_rate gauge",
            f'cache_hit_rate{{company_id="{self.company_id}"}} {self._cache_hit_rate():.2f}',
            "",
            "# HELP ud_temporal_conflict_total Temporal conflicts detected (A1)",
            "# TYPE ud_temporal_conflict_total counter",
            f'ud_temporal_conflict_total{{company_id="{self.company_id}"}} {self.temporal_conflicts_count}',
            "",
            "# HELP ud_db_conflict_total Database conflicts detected (A2)",
            "# TYPE ud_db_conflict_total counter",
            f'ud_db_conflict_total{{company_id="{self.company_id}"}} {self.db_conflicts_count}',
            "",
            "# HELP ud_osrm_cache_hit_rate OSRM cache hit rate (A5)",
            "# TYPE ud_osrm_cache_hit_rate gauge",
            f'ud_osrm_cache_hit_rate{{company_id="{self.company_id}"}} {self._osrm_cache_hit_rate():.2f}',
            "",
            "# HELP ud_osrm_cache_bypass_total OSRM cache bypasses (Redis HS) (A5)",
            "# TYPE ud_osrm_cache_bypass_total counter",
            f'ud_osrm_cache_bypass_total{{company_id="{self.company_id}"}} {self.osrm_cache_bypass_count}',
        ]
        return "\n".join(lines)

    def _cache_hit_rate(self) -> float:
        """Calcule le taux de succès du cache."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total
    
    def _osrm_cache_hit_rate(self) -> float:
        """Calcule le taux de succès du cache OSRM (A5)."""
        total = self.osrm_cache_hits + self.osrm_cache_misses
        if total == 0:
            return 0.0
        return self.osrm_cache_hits / total


class DispatchMetricsCollector(object):
    """Collecteur de métriques de performance pour un dispatch."""

    def __init__(self, company_id: int, dispatch_run_id: int | None = None):
        """Initialise le collecteur.

        Args:
            company_id: ID de l'entreprise
            dispatch_run_id: ID du dispatch run (optionnel)
        """
        super().__init__()
        self.company_id = company_id
        self.dispatch_run_id = dispatch_run_id
        self.metrics = DispatchPerformanceMetrics(
            dispatch_run_id=dispatch_run_id,
            company_id=company_id
        )
        self._timers: Dict[str, float] = {}
        self._start_times: Dict[str, float] = {}

    @contextmanager
    def time_step(self, step_name: str):
        """Context manager pour mesurer le temps d'une étape.

        Args:
            step_name: Nom de l'étape (ex: 'data_collection', 'heuristics')
        """
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self._timers[step_name] = elapsed
            logger.debug("[PerformanceMetrics] %s: %.3fs", step_name, elapsed)

    def start_timer(self, step_name: str) -> None:
        """Démarre un timer pour une étape.

        Args:
            step_name: Nom de l'étape
        """
        self._start_times[step_name] = time.time()

    def end_timer(self, step_name: str) -> float:
        """Arrête un timer et retourne le temps écoulé.

        Args:
            step_name: Nom de l'étape
        Returns:
            Temps écoulé en secondes
        """
        if step_name not in self._start_times:
            logger.warning("[PerformanceMetrics] Timer %s non démarré", step_name)
            return 0.0

        elapsed = time.time() - self._start_times[step_name]
        self._timers[step_name] = elapsed
        del self._start_times[step_name]
        return elapsed

    def increment_counter(self, counter_name: str, value: int = 1) -> None:
        """Incrémente un compteur.

        Args:
            counter_name: Nom du compteur (ex: 'sql_queries', 'cache_hits')
            value: Valeur à ajouter (défaut: 1)
        """
        attr_name = f"{counter_name}_count" if not counter_name.endswith("_count") else counter_name

        if hasattr(self.metrics, attr_name):
            current = getattr(self.metrics, attr_name)
            setattr(self.metrics, attr_name, current + value)
        else:
            logger.warning("[PerformanceMetrics] Compteur %s inconnu", counter_name)

    def set_counter(self, counter_name: str, value: int) -> None:
        """Définit la valeur d'un compteur.

        Args:
            counter_name: Nom du compteur
            value: Nouvelle valeur
        """
        attr_name = f"{counter_name}_count" if not counter_name.endswith("_count") else counter_name

        if hasattr(self.metrics, attr_name):
            setattr(self.metrics, attr_name, value)
        else:
            logger.warning("[PerformanceMetrics] Compteur %s inconnu", counter_name)

    def finalize(self, algorithm_used: str = "auto", feature_flags: Dict[str, bool] | None = None) -> DispatchPerformanceMetrics:
        """Finalise les métriques et retourne le résultat.

        Args:
            algorithm_used: Algorithme utilisé ('heuristics', 'solver', 'auto')
            feature_flags: Dict des feature flags actifs
        Returns:
            DispatchPerformanceMetrics finalisé
        """
        # Appliquer les timers aux attributs
        if "data_collection" in self._timers:
            self.metrics.data_collection_time = self._timers["data_collection"]
        if "heuristics" in self._timers:
            self.metrics.heuristics_time = self._timers["heuristics"]
        if "solver" in self._timers:
            self.metrics.solver_time = self._timers["solver"]
        if "persistence" in self._timers:
            self.metrics.persistence_time = self._timers["persistence"]

        # Calculer le temps total
        self.metrics.total_time = sum(self._timers.values())

        # Métadonnées
        self.metrics.algorithm_used = algorithm_used
        if feature_flags:
            self.metrics.feature_flags = feature_flags

        # Log final
        logger.info(
            "[PerformanceMetrics] Dispatch %s terminé: %.3fs total (data=%.3fs, heur=%.3fs, solver=%.3fs, persist=%.3fs, SQL=%d, conflits=%d)",
            self.dispatch_run_id or "?", self.metrics.total_time,
            self.metrics.data_collection_time, self.metrics.heuristics_time,
            self.metrics.solver_time, self.metrics.persistence_time,
            self.metrics.sql_queries_count, self.metrics.temporal_conflicts_count
        )

        return self.metrics

    def get_metrics(self) -> DispatchPerformanceMetrics:
        """Retourne les métriques actuelles."""
        return self.metrics


# Instance globale pour tracking SQL queries (à intégrer avec SQLAlchemy events)
class SQLCounter(object):
    """Compteur SQL thread-safe."""
    _instance: 'SQLCounter | None' = None
    
    def __init__(self):
        super().__init__()
        self._counter = defaultdict(int)
    
    @classmethod
    def get_instance(cls) -> 'SQLCounter':
        """Retourne l'instance singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def reset(self) -> None:
        """Réinitialise le compteur."""
        self._counter.clear()
    
    def get_count(self) -> int:
        """Retourne le nombre total de requêtes."""
        return sum(self._counter.values())
    
    def increment(self, query_type: str = "SELECT") -> None:
        """Incrémente le compteur."""
        self._counter[query_type] += 1


def reset_sql_counter() -> None:
    """Réinitialise le compteur SQL."""
    SQLCounter.get_instance().reset()


def get_sql_count() -> int:
    """Retourne le nombre de requêtes SQL depuis le dernier reset."""
    return SQLCounter.get_instance().get_count()


def increment_sql_counter(query_type: str = "SELECT") -> None:
    """Incrémente le compteur SQL."""
    SQLCounter.get_instance().increment(query_type)
