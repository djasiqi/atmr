# backend/services/unified_dispatch/dispatch_prometheus_metrics.py
"""✅ Métriques Prometheus centralisées pour le dispatch.

Métriques exposées:
- dispatch_runs_total: Compteur de runs (par status, mode)
- dispatch_duration_seconds: Histogramme durée dispatch
- dispatch_quality_score: Gauge qualité (0-100)
- dispatch_assignment_rate: Gauge taux d'assignation (%)
- dispatch_unassigned_count: Gauge nombre de bookings non assignés
- dispatch_circuit_breaker_state: Gauge état circuit breaker OSRM (0=CLOSED, 1=OPEN, 2=HALF_OPEN)
- dispatch_temporal_conflicts_total: Compteur conflits temporels
- dispatch_db_conflicts_total: Compteur conflits DB
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

# Import optionnel prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Gauge = None
    Histogram = None

logger = logging.getLogger(__name__)

# ==================== Prometheus Metrics ====================

if PROMETHEUS_AVAILABLE and Counter is not None and Gauge is not None and Histogram is not None:
    # Compteur de runs dispatch
    DISPATCH_RUNS_TOTAL = Counter(
        "dispatch_runs_total",
        "Nombre total de runs dispatch",
        ["status", "mode", "company_id"],  # status: running, completed, failed
    )

    # Histogramme durée dispatch
    DISPATCH_DURATION_SECONDS = Histogram(
        "dispatch_duration_seconds",
        "Durée d'exécution du dispatch (secondes)",
        ["mode", "company_id"],
        buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    )

    # Gauge qualité dispatch
    DISPATCH_QUALITY_SCORE = Gauge(
        "dispatch_quality_score",
        "Score de qualité du dispatch (0-100)",
        ["dispatch_run_id", "company_id"],
    )

    # Gauge taux d'assignation
    DISPATCH_ASSIGNMENT_RATE = Gauge(
        "dispatch_assignment_rate",
        "Taux d'assignation des bookings (%)",
        ["dispatch_run_id", "company_id"],
    )

    # Gauge nombre de bookings non assignés
    DISPATCH_UNASSIGNED_COUNT = Gauge(
        "dispatch_unassigned_count",
        "Nombre de bookings non assignés",
        ["dispatch_run_id", "company_id"],
    )

    # Gauge état circuit breaker OSRM
    DISPATCH_CIRCUIT_BREAKER_STATE = Gauge(
        "dispatch_circuit_breaker_state",
        "État du circuit breaker OSRM (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
        ["company_id"],
    )

    # Compteur conflits temporels
    DISPATCH_TEMPORAL_CONFLICTS_TOTAL = Counter(
        "dispatch_temporal_conflicts_total",
        "Nombre total de conflits temporels détectés",
        ["dispatch_run_id", "company_id"],
    )

    # Compteur conflits DB
    DISPATCH_DB_CONFLICTS_TOTAL = Counter(
        "dispatch_db_conflicts_total",
        "Nombre total de conflits DB (contraintes uniques)",
        ["dispatch_run_id", "company_id"],
    )

    # Compteur OSRM cache hits/misses
    DISPATCH_OSRM_CACHE_HITS_TOTAL = Counter(
        "dispatch_osrm_cache_hits_total",
        "Nombre total de hits cache OSRM",
        ["dispatch_run_id", "company_id"],
    )

    DISPATCH_OSRM_CACHE_MISSES_TOTAL = Counter(
        "dispatch_osrm_cache_misses_total",
        "Nombre total de misses cache OSRM",
        ["dispatch_run_id", "company_id"],
    )
else:
    # Fallback si prometheus_client non disponible
    DISPATCH_RUNS_TOTAL = None
    DISPATCH_DURATION_SECONDS = None
    DISPATCH_QUALITY_SCORE = None
    DISPATCH_ASSIGNMENT_RATE = None
    DISPATCH_UNASSIGNED_COUNT = None
    DISPATCH_CIRCUIT_BREAKER_STATE = None
    DISPATCH_TEMPORAL_CONFLICTS_TOTAL = None
    DISPATCH_DB_CONFLICTS_TOTAL = None
    DISPATCH_OSRM_CACHE_HITS_TOTAL = None
    DISPATCH_OSRM_CACHE_MISSES_TOTAL = None


# ==================== Helper Functions ====================


def record_dispatch_run(
    status: str,
    mode: str,
    company_id: int,
) -> None:
    """Enregistre un run dispatch.

    Args:
        status: Status du run (running, completed, failed)
        mode: Mode du dispatch (auto, semi_auto, manual)
        company_id: ID de l'entreprise
    """
    if DISPATCH_RUNS_TOTAL:
        try:
            DISPATCH_RUNS_TOTAL.labels(
                status=status,
                mode=mode or "unknown",
                company_id=str(company_id),
            ).inc()
        except Exception as e:
            logger.warning("[DispatchMetrics] Erreur enregistrement run: %s", e)


def record_dispatch_duration(
    duration_seconds: float,
    mode: str,
    company_id: int,
) -> None:
    """Enregistre la durée d'un dispatch.

    Args:
        duration_seconds: Durée en secondes
        mode: Mode du dispatch
        company_id: ID de l'entreprise
    """
    if DISPATCH_DURATION_SECONDS:
        try:
            DISPATCH_DURATION_SECONDS.labels(
                mode=mode or "unknown",
                company_id=str(company_id),
            ).observe(duration_seconds)
        except Exception as e:
            logger.warning("[DispatchMetrics] Erreur enregistrement durée: %s", e)


def record_dispatch_quality(
    quality_score: float,
    dispatch_run_id: int,
    company_id: int,
) -> None:
    """Enregistre le score de qualité.

    Args:
        quality_score: Score de qualité (0-100)
        dispatch_run_id: ID du dispatch run
        company_id: ID de l'entreprise
    """
    if DISPATCH_QUALITY_SCORE:
        try:
            DISPATCH_QUALITY_SCORE.labels(
                dispatch_run_id=str(dispatch_run_id),
                company_id=str(company_id),
            ).set(quality_score)
        except Exception as e:
            logger.warning("[DispatchMetrics] Erreur enregistrement qualité: %s", e)


def record_assignment_rate(
    assignment_rate: float,
    dispatch_run_id: int,
    company_id: int,
) -> None:
    """Enregistre le taux d'assignation.

    Args:
        assignment_rate: Taux d'assignation (0-100)
        dispatch_run_id: ID du dispatch run
        company_id: ID de l'entreprise
    """
    if DISPATCH_ASSIGNMENT_RATE:
        try:
            DISPATCH_ASSIGNMENT_RATE.labels(
                dispatch_run_id=str(dispatch_run_id),
                company_id=str(company_id),
            ).set(assignment_rate)
        except Exception as e:
            logger.warning("[DispatchMetrics] Erreur enregistrement assignment rate: %s", e)


def record_unassigned_count(
    unassigned_count: int,
    dispatch_run_id: int,
    company_id: int,
) -> None:
    """Enregistre le nombre de bookings non assignés.

    Args:
        unassigned_count: Nombre de bookings non assignés
        dispatch_run_id: ID du dispatch run
        company_id: ID de l'entreprise
    """
    if DISPATCH_UNASSIGNED_COUNT:
        try:
            DISPATCH_UNASSIGNED_COUNT.labels(
                dispatch_run_id=str(dispatch_run_id),
                company_id=str(company_id),
            ).set(unassigned_count)
        except Exception as e:
            logger.warning("[DispatchMetrics] Erreur enregistrement unassigned count: %s", e)


def record_circuit_breaker_state(
    state: str,
    company_id: int,
) -> None:
    """Enregistre l'état du circuit breaker OSRM.

    Args:
        state: État (CLOSED=0, OPEN=1, HALF_OPEN=2)
        company_id: ID de l'entreprise
    """
    if DISPATCH_CIRCUIT_BREAKER_STATE:
        try:
            state_value = {"CLOSED": 0, "OPEN": 1, "HALF_OPEN": 2}.get(state, 0)
            DISPATCH_CIRCUIT_BREAKER_STATE.labels(
                company_id=str(company_id),
            ).set(state_value)
        except Exception as e:
            logger.warning("[DispatchMetrics] Erreur enregistrement circuit breaker: %s", e)


def record_temporal_conflicts(
    conflicts_count: int,
    dispatch_run_id: int,
    company_id: int,
) -> None:
    """Enregistre les conflits temporels.

    Args:
        conflicts_count: Nombre de conflits
        dispatch_run_id: ID du dispatch run
        company_id: ID de l'entreprise
    """
    if DISPATCH_TEMPORAL_CONFLICTS_TOTAL:
        try:
            DISPATCH_TEMPORAL_CONFLICTS_TOTAL.labels(
                dispatch_run_id=str(dispatch_run_id),
                company_id=str(company_id),
            ).inc(conflicts_count)
        except Exception as e:
            logger.warning("[DispatchMetrics] Erreur enregistrement conflits temporels: %s", e)


def record_db_conflicts(
    conflicts_count: int,
    dispatch_run_id: int,
    company_id: int,
) -> None:
    """Enregistre les conflits DB.

    Args:
        conflicts_count: Nombre de conflits
        dispatch_run_id: ID du dispatch run
        company_id: ID de l'entreprise
    """
    if DISPATCH_DB_CONFLICTS_TOTAL:
        try:
            DISPATCH_DB_CONFLICTS_TOTAL.labels(
                dispatch_run_id=str(dispatch_run_id),
                company_id=str(company_id),
            ).inc(conflicts_count)
        except Exception as e:
            logger.warning("[DispatchMetrics] Erreur enregistrement conflits DB: %s", e)


def record_osrm_cache_hit(
    dispatch_run_id: int | None,
    company_id: int,
) -> None:
    """Enregistre un hit cache OSRM.

    Args:
        dispatch_run_id: ID du dispatch run (optionnel)
        company_id: ID de l'entreprise
    """
    if DISPATCH_OSRM_CACHE_HITS_TOTAL:
        try:
            DISPATCH_OSRM_CACHE_HITS_TOTAL.labels(
                dispatch_run_id=str(dispatch_run_id) if dispatch_run_id else "unknown",
                company_id=str(company_id),
            ).inc()
        except Exception as e:
            logger.warning("[DispatchMetrics] Erreur enregistrement cache hit: %s", e)


def record_osrm_cache_miss(
    dispatch_run_id: int | None,
    company_id: int,
) -> None:
    """Enregistre un miss cache OSRM.

    Args:
        dispatch_run_id: ID du dispatch run (optionnel)
        company_id: ID de l'entreprise
    """
    if DISPATCH_OSRM_CACHE_MISSES_TOTAL:
        try:
            DISPATCH_OSRM_CACHE_MISSES_TOTAL.labels(
                dispatch_run_id=str(dispatch_run_id) if dispatch_run_id else "unknown",
                company_id=str(company_id),
            ).inc()
        except Exception as e:
            logger.warning("[DispatchMetrics] Erreur enregistrement cache miss: %s", e)


@contextmanager
def dispatch_metrics_context(
    dispatch_run_id: int | None,
    company_id: int,
    mode: str,
):
    """Context manager pour enregistrer automatiquement les métriques dispatch.

    Usage:
        with dispatch_metrics_context(dispatch_run_id, company_id, mode):
            # Code dispatch
            record_dispatch_quality(score, dispatch_run_id, company_id)

    Args:
        dispatch_run_id: ID du dispatch run (passé pour usage dans le contexte, non utilisé directement)
        company_id: ID de l'entreprise
        mode: Mode du dispatch

    Note:
        dispatch_run_id est accepté mais non utilisé directement dans cette fonction.
        Il est disponible pour être passé aux fonctions appelées dans le contexte (ex: record_dispatch_quality).
    """
    _ = dispatch_run_id  # Accepté mais non utilisé directement (utilisé par fonctions appelées dans le contexte)
    import time

    start_time = time.time()
    record_dispatch_run("running", mode, company_id)

    try:
        yield
        record_dispatch_run("completed", mode, company_id)
    except Exception:
        record_dispatch_run("failed", mode, company_id)
        raise
    finally:
        duration = time.time() - start_time
        record_dispatch_duration(duration, mode, company_id)
