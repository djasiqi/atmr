# backend/services/analytics/__init__.py
"""
Module d'analytics pour le système de dispatch.
Collecte, agrège et analyse les métriques de performance.
"""

from .metrics_collector import MetricsCollector, collect_dispatch_metrics
from .aggregator import MetricsAggregator, aggregate_daily_stats, get_period_analytics
from .insights import generate_insights

__all__ = [
    'MetricsCollector',
    'collect_dispatch_metrics',
    'MetricsAggregator',
    'aggregate_daily_stats',
    'get_period_analytics',
    'generate_insights',
]

