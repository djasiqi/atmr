"""
unified_dispatch.metrics - Métriques et Monitoring

Ce module contient :
- dispatch.py : Métriques du dispatch (ancien dispatch_metrics.py)
- prometheus.py : Métriques Prometheus (ancien dispatch_prometheus_metrics.py)
- performance.py : Métriques de performance (ancien performance_metrics.py)
- errors.py : Métriques d'erreurs (ancien error_metrics.py)
- osrm_cache.py : Métriques cache OSRM (ancien osrm_cache_metrics.py)
- slo.py : Suivi des SLO

Créé lors du refactoring B1 - 7 janvier 2025
"""

# Exports depuis dispatch.py pour compatibilité
from .dispatch import (
    DELAY_MINUTES_THRESHOLD,
    QUALITY_FORMULA_VERSION,
    QUALITY_THRESHOLD,
    QUALITY_WEIGHTS,
    DispatchMetricsCollector,
    DispatchQualityMetrics,
    collect_dispatch_metrics,
    get_quality_formula_hash,
)

# Exports publics
__all__ = [
    "DELAY_MINUTES_THRESHOLD",
    "QUALITY_FORMULA_VERSION",
    "QUALITY_THRESHOLD",
    "QUALITY_WEIGHTS",
    "DispatchMetricsCollector",
    "DispatchQualityMetrics",
    "collect_dispatch_metrics",
    "get_quality_formula_hash",
]
