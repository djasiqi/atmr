# backend/services/db_session_metrics.py
"""Métriques Prometheus pour le monitoring des sessions SQLAlchemy.

Ce module fournit des métriques pour surveiller l'utilisation des sessions DB :
- Nombre de transactions (commits, rollbacks)
- Durée des transactions
- Utilisation des context managers vs usage direct
- Erreurs de session
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

# Import optionnel prometheus_client (peut ne pas être installé en dev)
try:
    from prometheus_client import Counter, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None

logger = logging.getLogger(__name__)

# ==================== Prometheus Metrics ====================

# Métriques Prometheus pour sessions DB (créées uniquement si prometheus_client disponible)
if PROMETHEUS_AVAILABLE and Counter is not None and Histogram is not None:
    DB_TRANSACTION_TOTAL = Counter(
        "db_transaction_total",
        "Nombre total de transactions DB",
        ["operation"],  # operation: "commit", "rollback", "begin"
    )

    DB_TRANSACTION_DURATION = Histogram(
        "db_transaction_duration_seconds",
        "Durée des transactions DB (secondes)",
        ["operation"],  # operation: "commit", "rollback", "begin"
        buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    )

    DB_SESSION_ERRORS_TOTAL = Counter(
        "db_session_errors_total",
        "Nombre total d'erreurs de session DB",
        ["error_type"],  # error_type: "SQLAlchemyError", "IntegrityError", etc.
    )

    DB_CONTEXT_MANAGER_USAGE = Counter(
        "db_context_manager_usage_total",
        "Utilisation des context managers DB",
        ["manager_type"],  # manager_type: "db_transaction", "db_read_only", "db_batch_operation"
    )

    DB_DIRECT_SESSION_USAGE = Counter(
        "db_direct_session_usage_total",
        "Utilisation directe de db.session (sans context manager)",
        ["operation"],  # operation: "commit", "rollback", "add", "query"
    )

    # ✅ FIX: Initialiser les métriques avec 0.0 pour qu'elles apparaissent même si jamais incrémentées
    try:
        DB_TRANSACTION_TOTAL.labels(operation="commit").inc(0)
        DB_TRANSACTION_TOTAL.labels(operation="rollback").inc(0)
        DB_TRANSACTION_TOTAL.labels(operation="begin").inc(0)
        DB_CONTEXT_MANAGER_USAGE.labels(manager_type="db_transaction").inc(0)
        DB_CONTEXT_MANAGER_USAGE.labels(manager_type="db_read_only").inc(0)
        DB_CONTEXT_MANAGER_USAGE.labels(manager_type="db_batch_operation").inc(0)
        DB_DIRECT_SESSION_USAGE.labels(operation="commit").inc(0)
        DB_DIRECT_SESSION_USAGE.labels(operation="rollback").inc(0)
    except Exception as e:
        # Ignorer les erreurs d'initialisation (peut échouer si Prometheus non configuré)
        logger.debug("[DB Session Metrics] Failed to initialize metrics with 0.0: %s", e)
else:
    DB_TRANSACTION_TOTAL = None
    DB_TRANSACTION_DURATION = None
    DB_SESSION_ERRORS_TOTAL = None
    DB_CONTEXT_MANAGER_USAGE = None
    DB_DIRECT_SESSION_USAGE = None


@contextmanager
def track_transaction(operation: str) -> Any:
    """Context manager pour tracker une transaction DB.

    Args:
        operation: Type d'opération ("commit", "rollback", "begin")

    Yields:
        None (context manager)
    """
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        if DB_TRANSACTION_TOTAL is not None:
            DB_TRANSACTION_TOTAL.labels(operation=operation).inc()
        if DB_TRANSACTION_DURATION is not None:
            DB_TRANSACTION_DURATION.labels(operation=operation).observe(duration)


def track_context_manager_usage(manager_type: str) -> None:
    """Track l'utilisation d'un context manager DB.

    Args:
        manager_type: Type de context manager ("db_transaction", "db_read_only", "db_batch_operation")
    """
    if DB_CONTEXT_MANAGER_USAGE is not None:
        DB_CONTEXT_MANAGER_USAGE.labels(manager_type=manager_type).inc()


def track_direct_session_usage(operation: str) -> None:
    """Track l'utilisation directe de db.session (sans context manager).

    ⚠️ Cette fonction devrait être appelée pour détecter les usages directs
    qui devraient utiliser des context managers à la place.

    Args:
        operation: Type d'opération ("commit", "rollback", "add", "query")
    """
    if DB_DIRECT_SESSION_USAGE is not None:
        DB_DIRECT_SESSION_USAGE.labels(operation=operation).inc()
        # Logger un warning en mode DEBUG pour aider à identifier les usages à migrer
        logger.debug(
            "[DB Session Metrics] Usage direct de db.session.%s détecté. Considérer utiliser db_transaction() ou db_read_only() à la place.",
            operation,
        )


def track_session_error(error_type: str) -> None:
    """Track une erreur de session DB.

    Args:
        error_type: Type d'erreur ("SQLAlchemyError", "IntegrityError", etc.)
    """
    if DB_SESSION_ERRORS_TOTAL is not None:
        DB_SESSION_ERRORS_TOTAL.labels(error_type=error_type).inc()
