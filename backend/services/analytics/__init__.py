# backend/services/analytics/__init__.py
"""Module d'analytics pour le système de dispatch.
Collecte, agrège et analyse les métriques de performance.
"""

from .aggregator import MetricsAggregator, aggregate_daily_stats, get_period_analytics
from .insights import generate_insights
from .metrics_collector import MetricsCollector, collect_dispatch_metrics

__all__ = [
    "MetricsAggregator",
    "MetricsCollector",
    "aggregate_daily_stats",
    "collect_dispatch_metrics",
    "generate_insights",
    "get_period_analytics",
]
