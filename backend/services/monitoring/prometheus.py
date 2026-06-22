"""✅ 3.6.1: Service centralisé pour métriques Prometheus.

Centralise toutes les métriques Prometheus du système :
- WebSocket (connexions, déconnexions, latence)
- ETA (précision, latence calcul)
- OSRM (cache hit rate, latence)
- Dispatch (assignations/jour, retards)
- Localisation (positions/s, map-matching rate)
"""

import logging

try:
    from prometheus_client import (
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Gauge = None
    Histogram = None
    REGISTRY = None

logger = logging.getLogger(__name__)


def _get_or_create_metric(metric_class, name, *args, **kwargs):
    """Crée une métrique Prometheus ou retourne None si déjà enregistrée.

    Évite les erreurs de duplication lors d'imports multiples (Gunicorn workers).
    Si la métrique existe déjà, retourne None (elle sera utilisée depuis le
    registre global).

    Args:
        metric_class: Classe de métrique (Counter, Gauge, Histogram)
        name: Nom de la métrique
        *args, **kwargs: Arguments passés au constructeur de la métrique

    Returns:
        Instance de la métrique (nouvelle) ou None si déjà enregistrée
    """
    if not PROMETHEUS_AVAILABLE or REGISTRY is None:
        return None

    # Essayer de créer la métrique directement
    # Si elle existe déjà, Prometheus lèvera une ValueError
    try:
        return metric_class(name, *args, **kwargs)
    except ValueError as e:
        # Si la métrique existe déjà (duplication), logger et retourner None
        # La métrique existante sera utilisée depuis le registre global
        if "Duplicated timeseries" in str(e) or "already registered" in str(e):
            logger.debug(
                "[PrometheusMetrics] Métrique %s déjà enregistrée (ignorée, utilisation de l'existante)",
                name,
            )
            return None
        # Autre erreur : la propager
        raise


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
    # ✅ Protection contre duplication : utiliser _get_or_create_metric
    # Assignations par jour
    DISPATCH_ASSIGNMENTS_TOTAL = _get_or_create_metric(
        Counter,
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

# ==================== Push Notifications Metrics ====================

if PROMETHEUS_AVAILABLE and Counter and Histogram and Gauge:
    # Push notifications envoyées (succès/échec)
    PUSH_NOTIFICATIONS_TOTAL = Counter(
        "push_notifications_total",
        "Total push notifications envoyées",
        [
            "status",
            "event_type",
        ],  # status: "success", "failed", event_type: "booking", "message",
        # "delay", etc.
    )

    # Latence push notifications
    PUSH_NOTIFICATION_LATENCY_SECONDS = Histogram(
        "push_notification_latency_seconds",
        "Latence envoi push notification (secondes)",
        ["event_type"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    )

    # Retry count pour push notifications
    PUSH_NOTIFICATION_RETRIES_TOTAL = Counter(
        "push_notification_retries_total",
        "Total retries pour push notifications",
        ["event_type", "attempt"],  # attempt: "1", "2", "3", "4", "5"
    )

    # Taux de succès push notifications
    PUSH_NOTIFICATION_SUCCESS_RATE = Gauge(
        "push_notification_success_rate",
        "Taux de succès push notifications (0-1)",
        ["event_type"],
    )

    # ✅ INSTRUMENTATION: Tokens invalidés
    PUSH_TOKENS_INVALIDATED = Counter(
        "push_tokens_invalidated_total",
        "Total tokens invalidés",
        [
            "reason"
        ],  # "expired", "device_not_registered", "invalid_credentials", "logout"
    )

    # ✅ INSTRUMENTATION: Rate limit hits
    PUSH_RATE_LIMIT_HITS = Counter(
        "push_rate_limit_hits_total",
        "Total notifications bloquées par rate limit",
        ["driver_id"],
    )

    # P0.5: couverture enregistrement push (owners avec ≥1 token actif)
    PUSH_ACTIVE_OWNERS = Gauge(
        "push_active_owners_total",
        "Propriétaires avec au moins un DeviceToken actif",
        ["owner_type"],
    )

    PUSH_OPERATIONAL_DRIVERS_TOTAL = Gauge(
        "push_operational_drivers_total",
        "Chauffeurs opérationnels (is_active et is_available)",
    )

    PUSH_OPERATIONAL_DRIVERS_WITH_ACTIVE_TOKEN_TOTAL = Gauge(
        "push_operational_drivers_with_active_token_total",
        "Chauffeurs opérationnels avec au moins un token push actif",
    )

    PUSH_OPERATIONAL_DRIVERS_WITHOUT_ACTIVE_TOKEN_TOTAL = Gauge(
        "push_operational_drivers_without_active_token_total",
        "Chauffeurs opérationnels sans token push actif",
    )

    PUSH_TOKEN_REGISTRATION_SUCCESS_TOTAL = Counter(
        "push_token_registration_success_total",
        "Enregistrements push token réussis (save-push-token)",
        ["owner_type", "provider", "platform"],
    )

    PUSH_TOKEN_REGISTRATION_FAILURE_TOTAL = Counter(
        "push_token_registration_failure_total",
        "Échecs enregistrement push token (save-push-token)",
        ["owner_type", "provider", "platform", "reason"],
    )

    DRIVER_PUSH_CHANNEL_TOTAL = Counter(
        "driver_push_channel_total",
        "Push chauffeur envoyés par canal Android",
        ["channel"],
    )

    DRIVER_PUSH_SKIPPED_TOTAL = Counter(
        "driver_push_skipped_total",
        "Push chauffeur ignorés par raison métier",
        ["reason"],
    )

    COMPANY_PUSH_NEW_REQUEST_SENT_TOTAL = Counter(
        "company_push_new_request_sent_total",
        "Push new_request institution enqueue vers entreprise",
        ["company_id"],
    )

    COMPANY_PUSH_NEW_REQUEST_DELIVERY_FAILED_TOTAL = Counter(
        "company_push_new_request_delivery_failed_total",
        "Échec livraison push new_request entreprise",
        ["reason"],
    )

    COMPANY_PUSH_NEW_REQUEST_OPENED_TOTAL = Counter(
        "company_push_new_request_opened_total",
        "Ouverture notification new_request (telemetry mobile)",
        [],
    )

    COMPANY_PUSH_NEW_REQUEST_ACCEPT_TOTAL = Counter(
        "company_push_new_request_accept_total",
        "Acceptation offre institution après push mobile",
        [],
    )

    COMPANY_PUSH_NEW_REQUEST_REJECT_TOTAL = Counter(
        "company_push_new_request_reject_total",
        "Refus offre institution après notification",
        [],
    )

    COMPANY_PUSH_NEW_REQUEST_EXPIRED_TOTAL = Counter(
        "company_push_new_request_expired_total",
        "Tentative accept sur offre expirée",
        [],
    )

    COMPANY_PUSH_OPEN_TO_ACCEPT_SECONDS = Histogram(
        "company_push_open_to_accept_seconds",
        "Délai entre ouverture notif et acceptation offre",
        buckets=[5, 15, 30, 60, 120, 300, 600, 1800, 3600],
    )

    COMPANY_PUSH_TAP_WITHOUT_NETWORK_TOTAL = Counter(
        "company_push_new_request_tap_without_network_total",
        "Tap notification/offre institution sans réseau",
        [],
    )
else:
    PUSH_NOTIFICATIONS_TOTAL = None
    PUSH_NOTIFICATION_LATENCY_SECONDS = None
    PUSH_NOTIFICATION_RETRIES_TOTAL = None
    PUSH_NOTIFICATION_SUCCESS_RATE = None
    PUSH_TOKENS_INVALIDATED = None
    PUSH_RATE_LIMIT_HITS = None
    PUSH_ACTIVE_OWNERS = None
    PUSH_OPERATIONAL_DRIVERS_TOTAL = None
    PUSH_OPERATIONAL_DRIVERS_WITH_ACTIVE_TOKEN_TOTAL = None
    PUSH_OPERATIONAL_DRIVERS_WITHOUT_ACTIVE_TOKEN_TOTAL = None
    PUSH_TOKEN_REGISTRATION_SUCCESS_TOTAL = None
    PUSH_TOKEN_REGISTRATION_FAILURE_TOTAL = None
    DRIVER_PUSH_CHANNEL_TOTAL = None
    DRIVER_PUSH_SKIPPED_TOTAL = None
    COMPANY_PUSH_NEW_REQUEST_SENT_TOTAL = None
    COMPANY_PUSH_NEW_REQUEST_DELIVERY_FAILED_TOTAL = None
    COMPANY_PUSH_NEW_REQUEST_OPENED_TOTAL = None
    COMPANY_PUSH_NEW_REQUEST_ACCEPT_TOTAL = None
    COMPANY_PUSH_NEW_REQUEST_REJECT_TOTAL = None
    COMPANY_PUSH_NEW_REQUEST_EXPIRED_TOTAL = None
    COMPANY_PUSH_OPEN_TO_ACCEPT_SECONDS = None

# ==================== Resync Metrics ====================

if PROMETHEUS_AVAILABLE and Counter and Histogram and Gauge:
    # Resync déclenchés
    RESYNC_TOTAL = Counter(
        "resync_total",
        "Total resync déclenchés",
        ["type", "platform"],  # type: "bookings", "messages", platform: "mobile", "web"
    )

    # Durée resync
    RESYNC_DURATION_SECONDS = Histogram(
        "resync_duration_seconds",
        "Durée resync (secondes)",
        ["type", "platform"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )

    # Volume de données resynchronisées
    RESYNC_DATA_VOLUME = Histogram(
        "resync_data_volume",
        "Volume de données resynchronisées (nombre d'items)",
        ["type", "platform"],
        buckets=[1, 5, 10, 50, 100, 500, 1000],
    )

    # Fréquence resync (temps depuis dernière sync)
    RESYNC_INTERVAL_SECONDS = Histogram(
        "resync_interval_seconds",
        "Intervalle entre resyncs (secondes)",
        ["type", "platform"],
        buckets=[60, 300, 600, 1800, 3600, 7200, 14400],  # 1min à 4h
    )
else:
    RESYNC_TOTAL = None
    RESYNC_DURATION_SECONDS = None
    RESYNC_DATA_VOLUME = None
    RESYNC_INTERVAL_SECONDS = None


# ==================== Driver Mobile 2G/3G KPI (Phase 9) ====================

if PROMETHEUS_AVAILABLE and Counter and Histogram:
    DRIVER_MOBILE_SNAPSHOT_REQUESTS = _get_or_create_metric(
        Counter,
        "driver_mobile_snapshot_requests_total",
        "Total requêtes snapshot mobile chauffeur",
        ["outcome"],  # outcome: "success", "error"
    )

    DRIVER_BOOKING_STATUS_UPDATES = _get_or_create_metric(
        Counter,
        "driver_booking_status_updates_total",
        "Total mises à jour statut mission (idempotency)",
        ["idempotency_status"],  # idempotency_status: "new", "replay", "conflict"
    )
else:
    DRIVER_MOBILE_SNAPSHOT_REQUESTS = None
    DRIVER_BOOKING_STATUS_UPDATES = None


# ==================== Helper Functions ====================


def track_eta_calculation(
    source: str,
    latency_seconds: float,
    accuracy: float | None = None,
    zone: str | None = None,
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


# ==================== Push Notifications Helper Functions ====================


def track_push_notification(
    status: str,
    event_type: str,
    latency_seconds: float | None = None,
    attempts: int = 1,
) -> None:
    """Enregistre une push notification.

    Args:
        status: Statut ("success", "failed")
        event_type: Type d'événement ("booking", "message", "delay", "alert", etc.)
        latency_seconds: Latence en secondes (optionnel)
        attempts: Nombre de tentatives (défaut: 1)
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        if PUSH_NOTIFICATIONS_TOTAL:
            PUSH_NOTIFICATIONS_TOTAL.labels(status=status, event_type=event_type).inc()

        if latency_seconds is not None and PUSH_NOTIFICATION_LATENCY_SECONDS:
            PUSH_NOTIFICATION_LATENCY_SECONDS.labels(event_type=event_type).observe(
                latency_seconds
            )

        if attempts > 1 and PUSH_NOTIFICATION_RETRIES_TOTAL:
            # Enregistrer chaque retry
            for attempt_num in range(2, attempts + 1):
                PUSH_NOTIFICATION_RETRIES_TOTAL.labels(
                    event_type=event_type, attempt=str(attempt_num)
                ).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking push notification: %s", e)


def track_push_token_invalidated(reason: str) -> None:
    """Enregistre l'invalidation d'un token push.

    Args:
        reason: Raison de l'invalidation ("expired", "device_not_registered",
            "invalid_credentials", "logout", etc.)
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        if PUSH_TOKENS_INVALIDATED:
            PUSH_TOKENS_INVALIDATED.labels(reason=reason).inc()
    except Exception as e:
        logger.debug(
            "[PrometheusMetrics] Error tracking push token invalidation: %s", e
        )


def refresh_push_active_owners_gauges() -> None:
    """Met à jour les gauges de couverture push (drivers / companies / opérationnels)."""
    if not PROMETHEUS_AVAILABLE or not PUSH_ACTIVE_OWNERS:
        return

    try:
        from sqlalchemy import distinct, func

        from models import DeviceToken, Driver

        driver_owners = (
            DeviceToken.query.with_entities(func.count(distinct(DeviceToken.driver_id)))
            .filter(
                DeviceToken.driver_id.isnot(None),
                DeviceToken.is_active.is_(True),
            )
            .scalar()
        )
        company_owners = (
            DeviceToken.query.with_entities(
                func.count(distinct(DeviceToken.company_id))
            )
            .filter(
                DeviceToken.company_id.isnot(None),
                DeviceToken.is_active.is_(True),
            )
            .scalar()
        )
        PUSH_ACTIVE_OWNERS.labels(owner_type="driver").set(int(driver_owners or 0))
        PUSH_ACTIVE_OWNERS.labels(owner_type="company").set(int(company_owners or 0))

        operational_filter = (
            Driver.is_active.is_(True),
            Driver.is_available.is_(True),
        )
        operational_total = (
            Driver.query.with_entities(func.count(Driver.id))
            .filter(*operational_filter)
            .scalar()
        )

        operational_with_token = (
            Driver.query.with_entities(func.count(func.distinct(Driver.id)))
            .join(DeviceToken, DeviceToken.driver_id == Driver.id)
            .filter(
                *operational_filter,
                DeviceToken.is_active.is_(True),
            )
            .scalar()
        )

        op_total = int(operational_total or 0)
        op_with = int(operational_with_token or 0)
        op_without = max(op_total - op_with, 0)

        if PUSH_OPERATIONAL_DRIVERS_TOTAL:
            PUSH_OPERATIONAL_DRIVERS_TOTAL.set(op_total)
        if PUSH_OPERATIONAL_DRIVERS_WITH_ACTIVE_TOKEN_TOTAL:
            PUSH_OPERATIONAL_DRIVERS_WITH_ACTIVE_TOKEN_TOTAL.set(op_with)
        if PUSH_OPERATIONAL_DRIVERS_WITHOUT_ACTIVE_TOKEN_TOTAL:
            PUSH_OPERATIONAL_DRIVERS_WITHOUT_ACTIVE_TOKEN_TOTAL.set(op_without)
    except Exception as e:
        logger.debug(
            "[PrometheusMetrics] Error refreshing push active owners: %s", e
        )


def _normalize_push_registration_labels(
    payload: dict[str, object] | None,
) -> tuple[str, str]:
    raw = payload or {}
    provider = str(raw.get("provider") or "expo").lower()
    platform = str(raw.get("platform") or "unknown").lower()
    if platform not in ("ios", "android"):
        platform = "unknown"
    if provider not in ("expo", "fcm"):
        provider = "expo"
    return provider, platform


def _registration_failure_reason(status_code: int) -> str:
    if status_code in (401, 403, 404):
        return "auth"
    if status_code >= 500:
        return "server"
    return "validation"


def track_push_token_registration_outcome(
    *,
    owner_type: str,
    status_code: int,
    payload: dict[str, object] | None = None,
) -> None:
    """Compteurs success/failure pour save-push-token (régression release mobile)."""
    if not PROMETHEUS_AVAILABLE:
        return
    provider, platform = _normalize_push_registration_labels(payload)
    try:
        if 200 <= status_code < 300 and PUSH_TOKEN_REGISTRATION_SUCCESS_TOTAL:
            PUSH_TOKEN_REGISTRATION_SUCCESS_TOTAL.labels(
                owner_type=owner_type,
                provider=provider,
                platform=platform,
            ).inc()
        elif PUSH_TOKEN_REGISTRATION_FAILURE_TOTAL:
            PUSH_TOKEN_REGISTRATION_FAILURE_TOTAL.labels(
                owner_type=owner_type,
                provider=provider,
                platform=platform,
                reason=_registration_failure_reason(status_code),
            ).inc()
    except Exception as e:
        logger.debug(
            "[PrometheusMetrics] Error tracking push token registration: %s", e
        )


def track_push_rate_limit_hit(driver_id: int | None) -> None:
    """Enregistre un hit de rate limit pour push notifications.

    Args:
        driver_id: ID du driver (ou None si non disponible)
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        if PUSH_RATE_LIMIT_HITS:
            driver_id_str = str(driver_id) if driver_id is not None else "unknown"
            PUSH_RATE_LIMIT_HITS.labels(driver_id=driver_id_str).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking push rate limit hit: %s", e)


def update_push_notification_success_rate(event_type: str, rate: float) -> None:
    """Met à jour le taux de succès push notifications.

    Args:
        event_type: Type d'événement
        rate: Taux de succès (0-1)
    """
    if not PROMETHEUS_AVAILABLE or not PUSH_NOTIFICATION_SUCCESS_RATE:
        return

    try:
        PUSH_NOTIFICATION_SUCCESS_RATE.labels(event_type=event_type).set(rate)
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error updating push success rate: %s", e)


# ==================== Resync Helper Functions ====================


def track_driver_mobile_snapshot(outcome: str = "success") -> None:
    """Plan 2G/3G Phase 9 : KPI snapshot mobile."""
    if not PROMETHEUS_AVAILABLE or not DRIVER_MOBILE_SNAPSHOT_REQUESTS:
        return
    try:
        DRIVER_MOBILE_SNAPSHOT_REQUESTS.labels(outcome=outcome).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking snapshot: %s", e)


def track_driver_booking_status_update(idempotency_status: str) -> None:
    """Plan 2G/3G Phase 9 : KPI mises à jour statut (new, replay, conflict)."""
    if not PROMETHEUS_AVAILABLE or not DRIVER_BOOKING_STATUS_UPDATES:
        return
    try:
        DRIVER_BOOKING_STATUS_UPDATES.labels(
            idempotency_status=idempotency_status
        ).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking status update: %s", e)


def track_resync(
    resync_type: str,
    platform: str,
    duration_seconds: float,
    data_volume: int,
    interval_seconds: float | None = None,
) -> None:
    """Enregistre un resync.

    Args:
        resync_type: Type de resync ("bookings", "messages")
        platform: Plateforme ("mobile", "web")
        duration_seconds: Durée du resync en secondes
        data_volume: Volume de données récupérées (nombre d'items)
        interval_seconds: Intervalle depuis le dernier resync en secondes (optionnel)
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        if RESYNC_TOTAL:
            RESYNC_TOTAL.labels(type=resync_type, platform=platform).inc()

        if RESYNC_DURATION_SECONDS:
            RESYNC_DURATION_SECONDS.labels(type=resync_type, platform=platform).observe(
                duration_seconds
            )

        if RESYNC_DATA_VOLUME:
            RESYNC_DATA_VOLUME.labels(type=resync_type, platform=platform).observe(
                data_volume
            )

        if interval_seconds is not None and RESYNC_INTERVAL_SECONDS:
            RESYNC_INTERVAL_SECONDS.labels(type=resync_type, platform=platform).observe(
                interval_seconds
            )
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking resync: %s", e)


# ==================== Invoice PDF Metrics ====================

if PROMETHEUS_AVAILABLE and Histogram and Counter:
    # Durée de génération PDF (ms)
    INVOICE_PDF_GENERATION_MS = _get_or_create_metric(
        Histogram,
        "invoice_pdf_generation_ms",
        "Durée de génération PDF facture/rappel (millisecondes)",
        ["pdf_kind", "billing_type", "template_version"],
        buckets=[50, 100, 200, 400, 800, 1200, 1500, 2000, 3000, 5000, 8000],
    )

    # Nombre de lignes dans le PDF
    INVOICE_PDF_ROWS = _get_or_create_metric(
        Histogram,
        "invoice_pdf_rows",
        "Nombre de lignes dans le PDF (après regroupement aller/retour)",
        ["pdf_kind", "billing_type", "template_version"],
        buckets=[1, 5, 10, 20, 40, 60, 80, 120, 200],
    )

    # Warnings (seuils dépassés)
    INVOICE_PDF_WARNING_TOTAL = _get_or_create_metric(
        Counter,
        "invoice_pdf_warning_total",
        "Total warnings génération PDF (seuils dépassés)",
        ["reason", "pdf_kind", "billing_type", "template_version"],
    )
else:
    INVOICE_PDF_GENERATION_MS = None
    INVOICE_PDF_ROWS = None
    INVOICE_PDF_WARNING_TOTAL = None

# ==================== Booking audit / Notifications Kafka (PR-2 / PR-3) ====================

if PROMETHEUS_AVAILABLE and Counter and Histogram:
    BOOKING_AUDIT_WRITE_FAILED_TOTAL = _get_or_create_metric(
        Counter,
        "booking_audit_write_failed_total",
        "Échec persistance audit booking après succès métier",
        ["action_type"],
    )
    NOTIFICATION_KAFKA_SKIP_TOTAL = _get_or_create_metric(
        Counter,
        "notification_kafka_skip_total",
        "Messages notifications Kafka ignorés (skip + commit)",
        ["reason"],
    )
    NOTIFICATION_KAFKA_ENQUEUE_TOTAL = _get_or_create_metric(
        Counter,
        "notification_kafka_enqueue_total",
        "Tentatives publication push via Kafka depuis Celery",
        ["status"],
    )
    NOTIFICATION_KAFKA_ENQUEUE_LATENCY_SECONDS = _get_or_create_metric(
        Histogram,
        "notification_kafka_enqueue_latency_seconds",
        "Latence de la tentative d'enqueue Kafka (task send_push_via_kafka)",
        ["status"],
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
    )
else:
    BOOKING_AUDIT_WRITE_FAILED_TOTAL = None
    NOTIFICATION_KAFKA_SKIP_TOTAL = None
    NOTIFICATION_KAFKA_ENQUEUE_TOTAL = None
    NOTIFICATION_KAFKA_ENQUEUE_LATENCY_SECONDS = None


def inc_booking_audit_write_failed(*, action_type: str) -> None:
    """Incrémente l'échec d'écriture audit booking (PR-2)."""
    if not PROMETHEUS_AVAILABLE or not BOOKING_AUDIT_WRITE_FAILED_TOTAL:
        return
    try:
        BOOKING_AUDIT_WRITE_FAILED_TOTAL.labels(action_type=action_type).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] booking_audit_write_failed: %s", e)


def inc_notification_kafka_skip(*, reason: str) -> None:
    """Skip métier consumer notifications Kafka (PR-1)."""
    if not PROMETHEUS_AVAILABLE or not NOTIFICATION_KAFKA_SKIP_TOTAL:
        return
    try:
        NOTIFICATION_KAFKA_SKIP_TOTAL.labels(reason=reason).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] notification_kafka_skip: %s", e)


def inc_notification_kafka_enqueue(*, status: str) -> None:
    """Enqueue push via Kafka depuis la task (PR-3) — status success|fallback|exception."""
    if not PROMETHEUS_AVAILABLE or not NOTIFICATION_KAFKA_ENQUEUE_TOTAL:
        return
    try:
        NOTIFICATION_KAFKA_ENQUEUE_TOTAL.labels(status=status).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] notification_kafka_enqueue: %s", e)


def observe_notification_kafka_enqueue_latency(*, status: str, seconds: float) -> None:
    """Histogramme latence send_push_via_kafka (task) — status success|fallback|exception."""
    if not PROMETHEUS_AVAILABLE or not NOTIFICATION_KAFKA_ENQUEUE_LATENCY_SECONDS:
        return
    try:
        NOTIFICATION_KAFKA_ENQUEUE_LATENCY_SECONDS.labels(status=status).observe(
            seconds
        )
    except Exception as e:
        logger.debug("[PrometheusMetrics] notification_kafka_enqueue_latency: %s", e)


def inc_driver_push_channel(*, channel: str) -> None:
    """Incrémente le compteur push chauffeur par canal Android."""
    if not PROMETHEUS_AVAILABLE or not DRIVER_PUSH_CHANNEL_TOTAL:
        return
    try:
        DRIVER_PUSH_CHANNEL_TOTAL.labels(channel=channel or "unknown").inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] driver_push_channel: %s", e)


def inc_driver_push_skipped(*, reason: str) -> None:
    """Incrémente le compteur push chauffeur ignorés par raison métier."""
    if not PROMETHEUS_AVAILABLE or not DRIVER_PUSH_SKIPPED_TOTAL:
        return
    try:
        DRIVER_PUSH_SKIPPED_TOTAL.labels(reason=reason or "unknown").inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] driver_push_skipped: %s", e)


# ==================== Invoice PDF Helper Functions ====================


def observe_invoice_pdf_perf(
    pdf_kind: str,
    billing_type: str,
    template_version: str,
    nb_rows: int | None,
    duration_ms: int,
    warning_threshold_rows: int = 40,
    warning_threshold_ms: int = 1500,
) -> None:
    """Observe les métriques de performance pour la génération PDF.

    Args:
        pdf_kind: Type de PDF ("invoice" | "reminder")
        billing_type: Type de facturation ("client" | "clinic" | "partner" | "unknown")
        template_version: Version du template (ex: "unified_v1")
        nb_rows: Nombre de lignes après regroupement (None si non applicable)
        duration_ms: Durée de génération en millisecondes
        warning_threshold_rows: Seuil de warning pour nb_rows (défaut: 40)
        warning_threshold_ms: Seuil de warning pour duration_ms (défaut: 1500)
    """
    if not PROMETHEUS_AVAILABLE:
        return

    try:
        # Toujours observer la durée
        if INVOICE_PDF_GENERATION_MS:
            INVOICE_PDF_GENERATION_MS.labels(
                pdf_kind=pdf_kind,
                billing_type=billing_type,
                template_version=template_version,
            ).observe(duration_ms)

        # Observer le nombre de lignes si disponible
        if nb_rows is not None and INVOICE_PDF_ROWS:
            INVOICE_PDF_ROWS.labels(
                pdf_kind=pdf_kind,
                billing_type=billing_type,
                template_version=template_version,
            ).observe(nb_rows)

        # Incrémenter le counter de warning si seuils dépassés
        warnings = []
        if nb_rows is not None and nb_rows > warning_threshold_rows:
            warnings.append("rows")
        if duration_ms > warning_threshold_ms:
            warnings.append("time")

        if warnings and INVOICE_PDF_WARNING_TOTAL:
            reason = ",".join(warnings)
            INVOICE_PDF_WARNING_TOTAL.labels(
                reason=reason,
                pdf_kind=pdf_kind,
                billing_type=billing_type,
                template_version=template_version,
            ).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] Error tracking invoice PDF perf: %s", e)


def inc_company_push_new_request_sent(*, company_id: int) -> None:
    if not PROMETHEUS_AVAILABLE or not COMPANY_PUSH_NEW_REQUEST_SENT_TOTAL:
        return
    try:
        COMPANY_PUSH_NEW_REQUEST_SENT_TOTAL.labels(company_id=str(company_id)).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] company_push_new_request_sent: %s", e)


def inc_company_push_new_request_delivery_failed(*, reason: str) -> None:
    if not PROMETHEUS_AVAILABLE or not COMPANY_PUSH_NEW_REQUEST_DELIVERY_FAILED_TOTAL:
        return
    try:
        COMPANY_PUSH_NEW_REQUEST_DELIVERY_FAILED_TOTAL.labels(reason=reason).inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] company_push_delivery_failed: %s", e)


def inc_company_push_new_request_opened() -> None:
    if not PROMETHEUS_AVAILABLE or not COMPANY_PUSH_NEW_REQUEST_OPENED_TOTAL:
        return
    try:
        COMPANY_PUSH_NEW_REQUEST_OPENED_TOTAL.inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] company_push_opened: %s", e)


def inc_company_push_new_request_accept() -> None:
    if not PROMETHEUS_AVAILABLE or not COMPANY_PUSH_NEW_REQUEST_ACCEPT_TOTAL:
        return
    try:
        COMPANY_PUSH_NEW_REQUEST_ACCEPT_TOTAL.inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] company_push_accept: %s", e)


def inc_company_push_new_request_reject() -> None:
    if not PROMETHEUS_AVAILABLE or not COMPANY_PUSH_NEW_REQUEST_REJECT_TOTAL:
        return
    try:
        COMPANY_PUSH_NEW_REQUEST_REJECT_TOTAL.inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] company_push_reject: %s", e)


def inc_company_push_new_request_expired() -> None:
    if not PROMETHEUS_AVAILABLE or not COMPANY_PUSH_NEW_REQUEST_EXPIRED_TOTAL:
        return
    try:
        COMPANY_PUSH_NEW_REQUEST_EXPIRED_TOTAL.inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] company_push_expired: %s", e)


def observe_company_push_open_to_accept_seconds(*, seconds: float) -> None:
    if not PROMETHEUS_AVAILABLE or not COMPANY_PUSH_OPEN_TO_ACCEPT_SECONDS:
        return
    try:
        COMPANY_PUSH_OPEN_TO_ACCEPT_SECONDS.observe(max(0.0, float(seconds)))
    except Exception as e:
        logger.debug("[PrometheusMetrics] company_push_open_to_accept: %s", e)


def inc_company_push_tap_without_network() -> None:
    if not PROMETHEUS_AVAILABLE or not COMPANY_PUSH_TAP_WITHOUT_NETWORK_TOTAL:
        return
    try:
        COMPANY_PUSH_TAP_WITHOUT_NETWORK_TOTAL.inc()
    except Exception as e:
        logger.debug("[PrometheusMetrics] company_push_tap_without_network: %s", e)
