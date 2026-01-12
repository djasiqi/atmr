"""
⚠️ MODULE DE COMPATIBILITÉ - DÉPRÉCIÉ

Ce module est conservé pour la rétrocompatibilité uniquement.
Utilisez plutôt: from services.unified_dispatch.metrics.dispatch import ...

Créé lors du refactoring B1 - 7 janvier 2025
"""

# Réexporter tout depuis le nouveau module
from services.unified_dispatch.metrics.dispatch import (
    DELAY_MINUTES_THRESHOLD,
    QUALITY_FORMULA_VERSION,
    QUALITY_THRESHOLD,
    QUALITY_WEIGHTS,
    DispatchMetricsCollector,
    DispatchQualityMetrics,
    collect_dispatch_metrics,
    get_quality_formula_hash,
)

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
