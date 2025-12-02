"""✅ 3.6.1: Service centralisé pour métriques Prometheus.

Centralise toutes les métriques Prometheus du système :
- WebSocket (connexions, déconnexions, latence)
- ETA (précision, latence calcul)
- OSRM (cache hit rate, latence)
- Dispatch (assignations/jour, retards)
- Localisation (positions/s, map-matching rate)
"""

import logging
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Gauge = None
    Histogram = None

logger = logging.getLogger(__name__)

# ==================== WebSocket Metrics ====================
# (Déjà définies dans websocket_metrics.py, mais on les expose ici pour centralisation)

# ==================== ETA Metrics ====================

if PROMETHEUS_AVAILABLE and Counter and Gauge and Histogram:
    # Précision ETA
    ETA_ACCURACY_RATE = Gauge(
        "eta_accuracy_rate",
        "Taux de précision ETA (0-1)",
        ["zone"],  # zone: "city", "suburb", "highway"
    )

    # Latence calcul ETA
    ETA_CALCULATION_LATENCY_SECONDS = Histogram(
        "eta_calculation_latency_seconds",
        "Latence de calcul ETA (secondes)",
        ["source"],  # source: "osrm", "osrm_ml", "haversine"
        buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
    )

    # Compteurs ETA par source
    ETA_CALCULATIONS_TOTAL = Counter(
        "eta_calculations_total",
        "Total calculs ETA",
        ["source"],  # source: "osrm", "osrm_ml", "haversine", "haversine_adaptive"
    )

    # Erreurs ETA
    ETA_ERRORS_TOTAL = Counter(
        "eta_errors_total",
        "Total erreurs calcul ETA",
        ["error_type"],  # error_type: "osrm_timeout", "osrm_error", "invalid_coords"
    )
else:
    ETA_ACCURACY_RATE = None
    ETA_CALCULATION_LATENCY_SECONDS = None
    ETA_CALCULATIONS_TOTAL = None
    ETA_ERRORS_TOTAL = None

# ==================== Dispatch Metrics ====================
# (Déjà définies dans dispatch_metrics.py et performance_metrics.py)

if PROMETHEUS_AVAILABLE and Counter and Gauge:
    # Assignations par jour
    DISPATCH_ASSIGNMENTS_TOTAL = Counter(
        "dispatch_assignments_total",
        "Total assignations créées",
        ["company_id", "status"],  # status: "scheduled", "in_progress", "completed"
    )

    # Retards
    DISPATCH_DELAYS_TOTAL = Counter(
        "dispatch_delays_total",
        "Total retards détectés",
        ["company_id", "severity"],  # severity: "low", "medium", "high", "critical"
    )

    # Taux de retard
    DISPATCH_DELAY_RATE = Gauge(
        "dispatch_delay_rate",
        "Taux de retard (0-1)",
        ["company_id"],
    )

    # Temps résolution dispatch
    if Histogram is not None:
        DISPATCH_RESOLUTION_TIME_SECONDS = Histogram(
            "dispatch_resolution_time_seconds",
            "Temps de résolution dispatch (secondes)",
            ["company_id", "algorithm"],  # algorithm: "heuristic", "solver", "auto"
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )
    else:
        DISPATCH_RESOLUTION_TIME_SECONDS = None
else:
    DISPATCH_ASSIGNMENTS_TOTAL = None
    DISPATCH_DELAYS_TOTAL = None
    DISPATCH_DELAY_RATE = None
    DISPATCH_RESOLUTION_TIME_SECONDS = None

# ==================== Localisation Metrics ====================

if PROMETHEUS_AVAILABLE and Counter and Gauge and Histogram:
    # Positions par seconde
    LOCATION_POSITIONS_TOTAL = Counter(
        "location_positions_total",
        "Total positions GPS reçues",
        ["source"],  # source: "socketio", "http", "batch"
    )

    # Taux map-matching
    LOCATION_MAP_MATCHING_RATE = Gauge(
        "location_map_matching_rate",
        "Taux de map-matching réussi (0-1)",
    )

    # Latence traitement position
    LOCATION_PROCESSING_LATENCY_SECONDS = Histogram(
        "location_processing_latency_seconds",
        "Latence traitement position (secondes)",
        ["source"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    )

    # Géofencing
    GEOFENCING_ARRIVALS_TOTAL = Counter(
        "geofencing_arrivals_total",
        "Total arrivées détectées",
        ["type"],  # type: "pickup", "dropoff"
    )
else:
    LOCATION_POSITIONS_TOTAL = None
    LOCATION_MAP_MATCHING_RATE = None
    LOCATION_PROCESSING_LATENCY_SECONDS = None
    GEOFENCING_ARRIVALS_TOTAL = None

# ==================== Realtime Optimizer Metrics ====================

if PROMETHEUS_AVAILABLE and Histogram and Counter:
    # Temps d'exécution realtime optimizer
    REALTIME_OPTIMIZER_EXECUTION_TIME_SECONDS = Histogram(
        "realtime_optimizer_execution_time_seconds",
        "Temps d'exécution realtime optimizer (secondes)",
        ["company_id"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )

    # Opportunités détectées
    REALTIME_OPTIMIZER_OPPORTUNITIES_TOTAL = Counter(
        "realtime_optimizer_opportunities_total",
        "Total opportunités d'optimisation détectées",
        ["company_id", "severity"],  # severity: "low", "medium", "high", "critical"
    )

    # Réassignations automatiques
    REALTIME_OPTIMIZER_REASSIGNMENTS_TOTAL = Counter(
        "realtime_optimizer_reassignments_total",
        "Total réassignations automatiques",
        ["company_id"],
    )
else:
    REALTIME_OPTIMIZER_EXECUTION_TIME_SECONDS = None
    REALTIME_OPTIMIZER_OPPORTUNITIES_TOTAL = None
    REALTIME_OPTIMIZER_REASSIGNMENTS_TOTAL = None

# ==================== Driver Metrics ====================

if PROMETHEUS_AVAILABLE and Gauge:
    # Chauffeurs en ligne
    DRIVERS_ONLINE = Gauge(
        "drivers_online",
        "Nombre de chauffeurs en ligne",
        ["company_id"],
    )
else:
    DRIVERS_ONLINE = None


# ==================== Helper Functions ====================


def track_eta_calculation(
    source: str,
    latency_seconds: float,
    accuracy: Optional[float] = None,
    zone: Optional[str] = None,
) -> None:
    """Enregistre un calcul ETA.

    Args:
        source: Source du calcul ("osrm", "osrm_ml", "haversine", "haversine_adaptive")
        latency_seconds: Latence du calcul en secondes
        accuracy: Précision ETA (0-1, optionnel)
        zone: Zone géographique (optionnel)
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        if ETA_CALCULATIONS_TOTAL:
            ETA_CALCULATIONS_TOTAL.labels(source=source).inc()

        if ETA_CALCULATION_LATENCY_SECONDS is not None:
            ETA_CALCULATION_LATENCY_SECONDS.labels(source=source).observe(
                latency_seconds
            )

        if accuracy is not None and ETA_ACCURACY_RATE is not None:
            zone_label = zone or "unknown"
            ETA_ACCURACY_RATE.labels(zone=zone_label).set(accuracy)
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking ETA: %s", e)


def track_eta_error(error_type: str) -> None:
    """Enregistre une erreur ETA.

    Args:
        error_type: Type d'erreur ("osrm_timeout", "osrm_error", "invalid_coords")
    """
    if not PROMETHEUS_AVAILABLE or not ETA_ERRORS_TOTAL:
        return

    try:
        ETA_ERRORS_TOTAL.labels(error_type=error_type).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking ETA error: %s", e)


def track_dispatch_assignment(company_id: int, status: str = "scheduled") -> None:
    """Enregistre une assignation dispatch.

    Args:
        company_id: ID de l'entreprise
        status: Statut de l'assignation ("scheduled", "in_progress", "completed")
    """
    if not PROMETHEUS_AVAILABLE or not DISPATCH_ASSIGNMENTS_TOTAL:
        return

    try:
        DISPATCH_ASSIGNMENTS_TOTAL.labels(
            company_id=str(company_id), status=status
        ).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking dispatch assignment: %s", e)


def track_dispatch_delay(company_id: int, severity: str) -> None:
    """Enregistre un retard dispatch.

    Args:
        company_id: ID de l'entreprise
        severity: Sévérité ("low", "medium", "high", "critical")
    """
    if not PROMETHEUS_AVAILABLE or not DISPATCH_DELAYS_TOTAL:
        return

    try:
        DISPATCH_DELAYS_TOTAL.labels(
            company_id=str(company_id), severity=severity
        ).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking dispatch delay: %s", e)


def update_dispatch_delay_rate(company_id: int, rate: float) -> None:
    """Met à jour le taux de retard dispatch.

    Args:
        company_id: ID de l'entreprise
        rate: Taux de retard (0-1)
    """
    if not PROMETHEUS_AVAILABLE or not DISPATCH_DELAY_RATE:
        return

    try:
        DISPATCH_DELAY_RATE.labels(company_id=str(company_id)).set(rate)
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error updating delay rate: %s", e)


def track_location_position(source: str) -> None:
    """Enregistre une position GPS.

    Args:
        source: Source de la position ("socketio", "http", "batch")
    """
    if not PROMETHEUS_AVAILABLE or not LOCATION_POSITIONS_TOTAL:
        return

    try:
        LOCATION_POSITIONS_TOTAL.labels(source=source).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking location: %s", e)


def update_map_matching_rate(rate: float) -> None:
    """Met à jour le taux de map-matching.

    Args:
        rate: Taux de map-matching réussi (0-1)
    """
    if not PROMETHEUS_AVAILABLE or not LOCATION_MAP_MATCHING_RATE:
        return

    try:
        LOCATION_MAP_MATCHING_RATE.set(rate)
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error updating map matching rate: %s", e)


def track_location_processing(source: str, latency_seconds: float) -> None:
    """Enregistre la latence de traitement d'une position.

    Args:
        source: Source de la position
        latency_seconds: Latence en secondes
    """
    if not PROMETHEUS_AVAILABLE or not LOCATION_PROCESSING_LATENCY_SECONDS:
        return

    try:
        LOCATION_PROCESSING_LATENCY_SECONDS.labels(source=source).observe(
            latency_seconds
        )
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking location processing: %s", e)


def track_geofencing_arrival(arrival_type: str) -> None:
    """Enregistre une arrivée détectée par géofencing.

    Args:
        arrival_type: Type d'arrivée ("pickup", "dropoff")
    """
    if not PROMETHEUS_AVAILABLE or not GEOFENCING_ARRIVALS_TOTAL:
        return

    try:
        GEOFENCING_ARRIVALS_TOTAL.labels(type=arrival_type).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking geofencing: %s", e)


def track_realtime_optimizer_execution(
    company_id: int, execution_time_seconds: float
) -> None:
    """Enregistre le temps d'exécution du realtime optimizer.

    Args:
        company_id: ID de l'entreprise
        execution_time_seconds: Temps d'exécution en secondes
    """
    if not PROMETHEUS_AVAILABLE or not REALTIME_OPTIMIZER_EXECUTION_TIME_SECONDS:
        return

    try:
        REALTIME_OPTIMIZER_EXECUTION_TIME_SECONDS.labels(
            company_id=str(company_id)
        ).observe(execution_time_seconds)
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking realtime optimizer: %s", e)


def track_realtime_optimizer_opportunity(company_id: int, severity: str) -> None:
    """Enregistre une opportunité d'optimisation détectée.

    Args:
        company_id: ID de l'entreprise
        severity: Sévérité ("low", "medium", "high", "critical")
    """
    if not PROMETHEUS_AVAILABLE or not REALTIME_OPTIMIZER_OPPORTUNITIES_TOTAL:
        return

    try:
        REALTIME_OPTIMIZER_OPPORTUNITIES_TOTAL.labels(
            company_id=str(company_id), severity=severity
        ).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking optimizer opportunity: %s", e)


def track_realtime_optimizer_reassignment(company_id: int) -> None:
    """Enregistre une réassignation automatique.

    Args:
        company_id: ID de l'entreprise
    """
    if not PROMETHEUS_AVAILABLE or not REALTIME_OPTIMIZER_REASSIGNMENTS_TOTAL:
        return

    try:
        REALTIME_OPTIMIZER_REASSIGNMENTS_TOTAL.labels(company_id=str(company_id)).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking reassignment: %s", e)


def update_drivers_online(company_id: int, count: int) -> None:
    """Met à jour le nombre de chauffeurs en ligne.

    Args:
        company_id: ID de l'entreprise
        count: Nombre de chauffeurs en ligne
    """
    if not PROMETHEUS_AVAILABLE or not DRIVERS_ONLINE:
        return

    try:
        DRIVERS_ONLINE.labels(company_id=str(company_id)).set(count)
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error updating drivers online: %s", e)
