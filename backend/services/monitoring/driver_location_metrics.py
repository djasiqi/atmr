"""Métriques Prometheus — chaîne localisation chauffeur (PR1).

Labels bornés : ``location_mode`` normalisé backend ; ``accept_reason`` borné.
Désactivable via ``DRIVER_LOCATION_METRICS_ENABLED`` (défaut: true).
"""

from __future__ import annotations

import os

_KNOWN_ACCEPT_REASONS: frozenset[str] = frozenset(
    {
        "",
        "older_than_canonical",
        "too_old_for_mode",
        "accuracy_too_low",
        "redis_unavailable_no_arbitration",
        "invalid_payload",
        "cross_tenant_mismatch",
        "mission_live_missing_mission_id",
        "location_update_not_attempted",
        "duplicate_event_id",
        "duplicate_proximity",
    }
)


def _metrics_enabled() -> bool:
    return os.getenv("DRIVER_LOCATION_METRICS_ENABLED", "true").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _norm_reason(reason: str | None) -> str:
    if not reason:
        return ""
    r = str(reason).strip()
    if r in _KNOWN_ACCEPT_REASONS:
        return r
    return "_unknown"


# Plafond enregistrement skew (mobile vs backend) — au-delà : données suspectes / horloge cassée.
MAX_CLOCK_SKEW_RECORD_SEC = 172800.0  # 48 h


def _norm_mode(mode: str | None) -> str:
    """Aligné sur ``normalize_location_mode`` (valeurs invalides → ``mission_live``)."""
    m = (mode or "").strip()
    if m in ("mission_live", "availability_presence", "passive_last_known"):
        return m
    return "mission_live"


try:
    from prometheus_client import Counter, Gauge, Histogram
except ImportError:
    Counter = None
    Gauge = None
    Histogram = None

_RECEIVED = None
_INGESTED = None
_SOCKET_STALE_FALLBACK = None
_DEDUP_SKIPPED = None
_PROCESSED = None
_FANOUT = None
_FALLBACK_INDIVIDUAL = None
_CLOCK_SKEW = None
_BATCH_INGEST_SIZE = None
_PAYLOAD_LEGACY_LNG_USAGE = None
_TRACKING_KAFKA_PRODUCED = None
_TRACKING_KAFKA_PUBLISH_ERRORS = None
_TRACKING_KAFKA_DLQ = None
_TRACKING_KAFKA_REBALANCE = None
_TRACKING_KAFKA_CONSUMER_LAG = None
_TRACKING_PROCESSED_FANOUT_FAILURES = None
_TRACKING_KAFKA_E2E_LATENCY = None
_DRIVER_DEVICE_HEALTH_RECEIVED = None
_BATCH_RATE_LIMITED = None
_BATCH_POINTS_RECEIVED = None
_BATCH_POINTS_CANONICAL = None
_BATCH_POINTS_OBSERVABILITY = None
_BATCH_POINTS_SKIPPED = None
_TRACKING_ID_PROPAGATED = None
_TRACKING_ID_MISSING = None
_CANONICAL_REDIS_WRITE = None
_CANONICAL_OVERWRITE = None
_GPS_PROVIDER = None
_TRACKING_MISSION_LIVE_MISSING_MISSION_ID = None
_TRACKING_INVARIANT_VIOLATION = None
_TRACKING_POSITION_FRESHNESS = None
_TRACKING_DELIVERY_RESULT = None
_TRACKING_HTTP_ACCEPTED_ASYNC = None
_TRACKING_KAFKA_PERSIST = None
_TRACKING_INVALID_CONFIG = None

_VALID_TRANSPORTS = frozenset({"http", "socket", "socket_batch", "kafka"})


def _norm_transport(transport: str) -> str:
    t = (transport or "http").strip()
    return t if t in _VALID_TRANSPORTS else "http"

if Counter is not None:
    _DEDUP_SKIPPED = Counter(
        "driver_location_dedup_skipped_total",
        "Points ignorés avant persistance (idempotence ou proximité)",
        ["reason", "location_mode", "transport"],
    )
    _RECEIVED = Counter(
        "driver_location_received_total",
        "Positions reçues (HTTP ou socket), avant traitement complet",
        ["transport", "location_mode"],
    )
    # P0 Patch C — Sémantique explicite pour les dashboards :
    # - Ce compteur ne mesure PAS la persistance Redis, le fanout ni un succès métier bout-en-bout.
    # - Même instant et mêmes labels que driver_location_received_total via inc_received() :
    #   point compté au point d'entrée P0 uniquement s'il n'a pas été skippé par should_skip_location_ingest
    #   (chemins qui appliquent cette dédup). Les skips sont sur driver_location_dedup_skipped_total.
    _INGESTED = Counter(
        "driver_location_ingested_total",
        "P0: même sémantique que received_total (inc_received); pas pipeline complet",
        ["transport", "location_mode"],
    )
    _SOCKET_STALE_FALLBACK = Counter(
        "driver_location_socket_stale_fallback_total",
        "PUT HTTP avec en-tête socket-stale (socket connecté mais pipeline sans ACK récent)",
    )
    _PROCESSED = Counter(
        "driver_location_processed_total",
        "Positions traitées par LocationService",
        ["accept_status", "accept_reason", "location_mode", "transport"],
    )
    _FANOUT = Counter(
        "driver_location_fanout_events_total",
        "Événements Socket.IO émis (fanout entreprise)",
        ["event", "accept_status"],
    )
    _FALLBACK_INDIVIDUAL = Counter(
        "driver_location_batch_fallback_individual_total",
        "driver_location unitaire après échec batch côté client (flag batch_fallback)",
    )
    _PAYLOAD_LEGACY_LNG_USAGE = Counter(
        "company_realtime_payload_legacy_lng_usage_total",
        "Usage du champ legacy `lng` pour la normalisation payload realtime company",
        ["client_type", "event_name"],
    )
    _TRACKING_KAFKA_PRODUCED = Counter(
        "tracking_kafka_messages_produced_total",
        "Messages Kafka publiés par le pipeline tracking",
        ["topic"],
    )
    _TRACKING_KAFKA_PUBLISH_ERRORS = Counter(
        "tracking_kafka_publish_errors_total",
        "Erreurs de publication Kafka sur le pipeline tracking",
        ["topic", "stage"],
    )
    _TRACKING_KAFKA_DLQ = Counter(
        "tracking_kafka_dlq_messages_total",
        "Messages redirigés vers la DLQ tracking",
        ["reason"],
    )
    _TRACKING_KAFKA_DLQ_FORCE_COMMIT = Counter(
        "tracking_kafka_dlq_force_commit_total",
        "Offsets Kafka commités après échec DLQ (position GPS perdue)",
        ["reason"],
    )
    _TRACKING_KAFKA_REBALANCE = Counter(
        "tracking_kafka_rebalance_total",
        "Nombre de rebalances détectés sur le consumer tracking",
        ["event"],
    )
    _TRACKING_KAFKA_CONSUMER_LAG = Gauge(
        "tracking_kafka_consumer_lag",
        "Lag du consumer tracking (end_offset - position), agrégé par partition",
        ["group", "topic", "partition"],
    )
    _TRACKING_PROCESSED_FANOUT_FAILURES = Counter(
        "tracking_processed_fanout_failures_total",
        "Échecs traitement message dans le consumer driver.location.processed → fanout",
        ["error_type"],
    )
    _TRACKING_FANOUT_EMIT = Counter(
        "tracking_fanout_emit_total",
        "Émissions Socket.IO driver_location depuis un consumer Kafka processed",
        ["emitter"],
    )
    _DRIVER_DEVICE_HEALTH_RECEIVED = Counter(
        "driver_device_health_received_total",
        (
            "Heartbeats device-status reçus du mobile (canal santé app, séparé "
            "du GPS)"
        ),
        ["constraint_reason"],
    )
    _BATCH_RATE_LIMITED = Counter(
        "driver_location_batch_rate_limited_total",
        "Batches driver_location_batch rejetés par rate limiter WebSocket",
    )
    _BATCH_POINTS_RECEIVED = Counter(
        "driver_location_batch_points_received_total",
        "Points GPS dans batches acceptés (après filtre pipeline, avant traitement)",
        ["location_mode"],
    )
    _BATCH_POINTS_CANONICAL = Counter(
        "driver_location_batch_points_canonical_total",
        "Points batch aboutissant à accepted_canonical",
        ["location_mode"],
    )
    _BATCH_POINTS_OBSERVABILITY = Counter(
        "driver_location_batch_points_observability_total",
        "Points batch aboutissant à accepted_observability_only",
        ["location_mode"],
    )
    _BATCH_POINTS_SKIPPED = Counter(
        "driver_location_batch_points_skipped_total",
        "Points batch ignorés (dedup, validation, rejet)",
        ["reason", "location_mode"],
    )
    _TRACKING_ID_PROPAGATED = Counter(
        "driver_location_tracking_id_propagated_total",
        "Points batch avec tracking_event_id propagé jusqu'au fanout",
        ["transport"],
    )
    _TRACKING_ID_MISSING = Counter(
        "driver_location_tracking_id_missing_total",
        "Points batch sans tracking_event_id au fanout",
        ["transport"],
    )
    _CANONICAL_REDIS_WRITE = Counter(
        "driver_location_canonical_redis_write_total",
        "Écritures Redis driver:{id}:loc:canonical acceptées",
        ["location_mode", "transport"],
    )
    _CANONICAL_OVERWRITE = Counter(
        "driver_location_canonical_overwrite_total",
        "Décisions d'arbitrage canonical (accepté vs rejeté plus ancien)",
        ["outcome", "location_mode"],
    )
    _GPS_PROVIDER = Counter(
        "driver_location_gps_provider_total",
        "Provider GPS déclaré dans le payload chauffeur",
        ["provider", "platform"],
    )
    _TRACKING_MISSION_LIVE_MISSING_MISSION_ID = Counter(
        "tracking_mission_live_missing_mission_id_total",
        (
            "Payload mission_live reçu sans mission_id (P0-C gate — "
            "compteur post-déploiement, non cumulatif Redis)"
        ),
        ["transport", "action"],
    )
    _TRACKING_INVARIANT_VIOLATION = Counter(
        "tracking_invariant_violation_total",
        "Violation d'invariant architecture GPS (INV-1 à INV-8)",
        ["invariant_id", "company_id"],
    )
    _TRACKING_POSITION_FRESHNESS = Histogram(
        "driver_tracking_position_freshness_seconds",
        "Âge recorded_at → now à l'acceptation canonical (secondes)",
        ["company_id", "location_mode"],
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 15.0, 60.0, 300.0, 600.0),
    )
    _TRACKING_DELIVERY_RESULT = Counter(
        "tracking_delivery_result_total",
        "Résultat livraison position par mode, transport et issue",
        ["mode", "transport", "result"],
    )
    _TRACKING_HTTP_ACCEPTED_ASYNC = Counter(
        "tracking_http_accepted_async_total",
        "Réponses HTTP 202 (position mise en file Kafka, avant persist consumer)",
        ["location_mode"],
    )
    _TRACKING_KAFKA_PERSIST = Counter(
        "tracking_kafka_persist_total",
        "Traitements persist terminés dans ingest_consumer (labels finis)",
        ["accept_status"],
    )
    _TRACKING_INVALID_CONFIG = Counter(
        "tracking_invalid_config_total",
        "Démarrages refusés du consumer ingest (configuration incohérente)",
        ["reason"],
    )

if Histogram is not None:
    _CLOCK_SKEW = Histogram(
        "driver_location_clock_skew_seconds",
        "Écart absolu recorded_at (payload) vs réception backend (_store_location)",
        ["location_mode"],
        buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600, 1800, 3600],
    )
    _BATCH_INGEST_SIZE = Histogram(
        "driver_location_batch_ingest_size",
        "Nombre de points dans driver_location_batch après filtre pipeline",
        buckets=(1, 2, 3, 5, 7, 10, 15, 20, 30, 50),
    )
    _TRACKING_KAFKA_E2E_LATENCY = Histogram(
        "tracking_kafka_e2e_latency_seconds",
        "Latence E2E entre réception raw et publication processed",
        buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5),
    )
    _CANONICAL_UPDATE_LATENCY = Histogram(
        "driver_location_canonical_update_latency_seconds",
        "Délai client (sent/recorded) → écriture canonical acceptée (réception backend)",
        ["location_mode", "transport"],
        buckets=(0.5, 1, 2, 5, 10, 20, 60, 120, 300),
    )
    _CANONICAL_STALENESS_READ = Histogram(
        "driver_location_canonical_staleness_seconds",
        "Ancienneté last_seen à la lecture dispatch (liste entreprise)",
        ["location_mode"],
        buckets=(0.5, 1, 2, 5, 10, 20, 60, 120, 300, 600, 1200),
    )
    _BATCH_LATENCY = Histogram(
        "driver_location_batch_latency_seconds",
        "Durée traitement handler driver_location_batch (entrée → ACK)",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    _GPS_ACCURACY = Histogram(
        "driver_location_gps_accuracy_meters",
        "Précision GPS déclarée (accuracy) à l'ingestion",
        ["platform", "location_mode", "transport"],
        buckets=(1, 3, 5, 8, 12, 20, 30, 50, 75, 100, 150),
    )
    _GPS_SPEED = Histogram(
        "driver_location_gps_speed_kmh",
        "Vitesse GPS déclarée (m/s converti en km/h si besoin)",
        ["platform", "location_mode", "transport"],
        buckets=(0, 5, 15, 30, 50, 70, 90, 110, 130),
    )
    _GPS_HEADING = Histogram(
        "driver_location_gps_heading_deg",
        "Cap GPS déclaré (degrés)",
        ["platform", "location_mode", "transport"],
        buckets=(0, 45, 90, 135, 180, 225, 270, 315, 360),
    )
    _TRACKING_OSRM_REQUEST = Counter(
        "tracking_osrm_request_total",
        "Requêtes OSRM snap/map (LocationService)",
        ["operation", "result"],
    )
    _TRACKING_OSRM_LATENCY = Histogram(
        "tracking_osrm_latency_seconds",
        "Latence requêtes OSRM snap/map",
        ["operation"],
        buckets=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
    )
else:
    _CLOCK_SKEW = None
    _BATCH_INGEST_SIZE = None
    _TRACKING_KAFKA_E2E_LATENCY = None
    _CANONICAL_UPDATE_LATENCY = None
    _CANONICAL_STALENESS_READ = None
    _BATCH_LATENCY = None
    _GPS_ACCURACY = None
    _GPS_SPEED = None
    _GPS_HEADING = None
    _TRACKING_OSRM_REQUEST = None
    _TRACKING_OSRM_LATENCY = None


def inc_tracking_delivery_result(
    *,
    mode: str,
    transport: str,
    result: str,
) -> None:
    """Compteur STOP GATE PR1 — availability_presence/http/success, forbidden, etc."""
    if not _metrics_enabled() or _TRACKING_DELIVERY_RESULT is None:
        return
    lm = _norm_mode(mode)
    t = _norm_transport(transport)
    r = (
        result
        if result in ("success", "forbidden", "failure", "duplicate")
        else "_unknown"
    )
    _TRACKING_DELIVERY_RESULT.labels(mode=lm, transport=t, result=r).inc()


def inc_dedup_skipped(
    *,
    reason: str,
    location_mode: str,
    transport: str,
) -> None:
    if not _metrics_enabled() or _DEDUP_SKIPPED is None:
        return
    lm = _norm_mode(location_mode)
    t = _norm_transport(transport)
    r = (
        reason
        if reason in ("duplicate_event_id", "duplicate_proximity")
        else "_unknown"
    )
    _DEDUP_SKIPPED.labels(reason=r, location_mode=lm, transport=t).inc()


def inc_received(*, transport: str, location_mode: str) -> None:
    """Incrémente received + ingested (mêmes labels) pour les points non skippés en amont."""
    if not _metrics_enabled() or _RECEIVED is None:
        return
    lm = _norm_mode(location_mode)
    t = _norm_transport(transport)
    _RECEIVED.labels(transport=t, location_mode=lm).inc()
    if _INGESTED is not None:
        _INGESTED.labels(transport=t, location_mode=lm).inc()
    # Pont vers le compteur historique services/monitoring/prometheus.py (legacy dashboards)
    try:
        from services.monitoring.prometheus import track_location_position

        src = {"http": "http", "socket": "socketio", "socket_batch": "batch"}.get(
            t, "http"
        )
        track_location_position(src)
    except Exception:
        pass


def inc_socket_stale_fallback() -> None:
    """Mobile : repli HTTP alors que le socket semble OK mais sans succès pipeline récent."""
    if not _metrics_enabled() or _SOCKET_STALE_FALLBACK is None:
        return
    _SOCKET_STALE_FALLBACK.inc()


def inc_processed(
    *,
    accept_status: str,
    accept_reason: str | None,
    location_mode: str,
    transport: str,
) -> None:
    if not _metrics_enabled() or _PROCESSED is None:
        return
    lm = _norm_mode(location_mode)
    ar = _norm_reason(accept_reason)
    t = _norm_transport(transport)
    st = (
        accept_status
        if accept_status
        in (
            "accepted_canonical",
            "accepted_observability_only",
            "rejected_invalid",
            "skipped",
        )
        else "_unknown"
    )
    _PROCESSED.labels(
        accept_status=st,
        accept_reason=ar,
        location_mode=lm,
        transport=t,
    ).inc()


def inc_fanout(*, event: str, accept_status: str) -> None:
    if not _metrics_enabled() or _FANOUT is None:
        return
    ev = (
        event
        if event in ("driver_location_update", "driver_live_state_update")
        else "_unknown"
    )
    st = (
        accept_status
        if accept_status
        in (
            "accepted_canonical",
            "accepted_observability_only",
        )
        else "_unknown"
    )
    _FANOUT.labels(event=ev, accept_status=st).inc()


def observe_canonical_update_latency_seconds(
    *, location_mode: str, transport: str, seconds: float
) -> None:
    """Délai entre horodatage client et acceptation canonical (pipeline GPS)."""
    if not _metrics_enabled() or _CANONICAL_UPDATE_LATENCY is None:
        return
    lm = _norm_mode(location_mode)
    t = _norm_transport(transport)
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return
    if s < 0 or s > float(MAX_CLOCK_SKEW_RECORD_SEC):
        return
    _CANONICAL_UPDATE_LATENCY.labels(location_mode=lm, transport=t).observe(s)


def observe_canonical_staleness_seconds(
    *, location_mode: str, last_seen_seconds: float
) -> None:
    """Ancienneté de la position lue côté dispatch (dashboard)."""
    if not _metrics_enabled() or _CANONICAL_STALENESS_READ is None:
        return
    lm = _norm_mode(location_mode)
    try:
        sec = float(last_seen_seconds)
    except (TypeError, ValueError):
        return
    if sec < 0 or sec > 86400.0 * 2:
        return
    _CANONICAL_STALENESS_READ.labels(location_mode=lm).observe(sec)


def observe_clock_skew_seconds(*, location_mode: str, skew_seconds: float) -> None:
    """Horloge mobile vs backend — utile runbook skew (Prometheus / Grafana)."""
    if not _metrics_enabled() or _CLOCK_SKEW is None:
        return
    lm = _norm_mode(location_mode)
    try:
        s = float(skew_seconds)
    except (TypeError, ValueError):
        return
    if s < 0 or s > MAX_CLOCK_SKEW_RECORD_SEC:
        return
    _CLOCK_SKEW.labels(location_mode=lm).observe(s)


def observe_driver_location_batch_ingest_size(*, size: int) -> None:
    if not _metrics_enabled() or _BATCH_INGEST_SIZE is None:
        return
    try:
        n = int(size)
    except (TypeError, ValueError):
        return
    if n < 1:
        return
    _BATCH_INGEST_SIZE.observe(float(n))


def inc_batch_fallback_individual() -> None:
    if not _metrics_enabled() or _FALLBACK_INDIVIDUAL is None:
        return
    _FALLBACK_INDIVIDUAL.inc()


def inc_payload_legacy_lng_usage(*, client_type: str, event_name: str) -> None:
    if not _metrics_enabled() or _PAYLOAD_LEGACY_LNG_USAGE is None:
        return
    ct = client_type if client_type in ("mobile", "web") else "unknown"
    ev = (
        event_name
        if event_name in ("driver_location_update", "driver_live_state_update")
        else "_unknown"
    )
    _PAYLOAD_LEGACY_LNG_USAGE.labels(client_type=ct, event_name=ev).inc()


def inc_tracking_kafka_messages_produced(*, topic: str) -> None:
    if not _metrics_enabled() or _TRACKING_KAFKA_PRODUCED is None:
        return
    _TRACKING_KAFKA_PRODUCED.labels(topic=topic).inc()


def inc_tracking_kafka_publish_errors(*, topic: str, stage: str) -> None:
    if not _metrics_enabled() or _TRACKING_KAFKA_PUBLISH_ERRORS is None:
        return
    _TRACKING_KAFKA_PUBLISH_ERRORS.labels(topic=topic, stage=stage or "_unknown").inc()


def inc_tracking_kafka_dlq_messages(*, reason: str) -> None:
    if not _metrics_enabled() or _TRACKING_KAFKA_DLQ is None:
        return
    _TRACKING_KAFKA_DLQ.labels(reason=reason or "_unknown").inc()


def inc_tracking_kafka_dlq_force_commit(*, reason: str) -> None:
    if not _metrics_enabled() or _TRACKING_KAFKA_DLQ_FORCE_COMMIT is None:
        return
    r = (reason or "_unknown").strip() or "_unknown"
    if len(r) > 120:
        r = r[:120]
    _TRACKING_KAFKA_DLQ_FORCE_COMMIT.labels(reason=r).inc()


def observe_osrm_request(*, operation: str, result: str, duration_sec: float) -> None:
    if not _metrics_enabled():
        return
    op = operation if operation in ("nearest", "match") else "_unknown"
    res = result if result in ("success", "timeout", "error", "circuit_open") else "_unknown"
    if _TRACKING_OSRM_REQUEST is not None:
        _TRACKING_OSRM_REQUEST.labels(operation=op, result=res).inc()
    if _TRACKING_OSRM_LATENCY is not None and duration_sec >= 0 and res == "success":
        _TRACKING_OSRM_LATENCY.labels(operation=op).observe(float(duration_sec))


def inc_tracking_kafka_rebalance(*, event: str) -> None:
    if not _metrics_enabled() or _TRACKING_KAFKA_REBALANCE is None:
        return
    _TRACKING_KAFKA_REBALANCE.labels(event=event or "_unknown").inc()


def set_tracking_kafka_consumer_lag(
    *, group: str, topic: str, partition: int | str, lag: float
) -> None:
    """Positionne le lag consumer tracking pour une partition (P1-1a).

    ``lag = end_offset - position`` (lag « prêt à traiter », sans RPC ``committed()``).
    Sans effet si métriques désactivées ou prometheus_client absent.
    """
    if not _metrics_enabled() or _TRACKING_KAFKA_CONSUMER_LAG is None:
        return
    g = (group or "_unknown").strip() or "_unknown"
    t = (topic or "_unknown").strip() or "_unknown"
    _TRACKING_KAFKA_CONSUMER_LAG.labels(
        group=g[:120], topic=t[:120], partition=str(partition)
    ).set(max(0.0, float(lag)))


def inc_tracking_processed_fanout_failure(*, error_type: str) -> None:
    if not _metrics_enabled() or _TRACKING_PROCESSED_FANOUT_FAILURES is None:
        return
    et = (error_type or "_unknown").strip() or "_unknown"
    if len(et) > 120:
        et = et[:120]
    _TRACKING_PROCESSED_FANOUT_FAILURES.labels(error_type=et).inc()


def inc_tracking_fanout_emit(*, emitter: str) -> None:
    """Compteur d'émission fanout par émetteur (P1-2 : backend_fanout | ws_service)."""
    if not _metrics_enabled() or _TRACKING_FANOUT_EMIT is None:
        return
    em = (emitter or "_unknown").strip() or "_unknown"
    if em not in ("backend_fanout", "ws_service"):
        em = "_unknown"
    _TRACKING_FANOUT_EMIT.labels(emitter=em).inc()


def observe_tracking_kafka_e2e_latency(*, latency_ms: float) -> None:
    if not _metrics_enabled() or _TRACKING_KAFKA_E2E_LATENCY is None:
        return
    if latency_ms < 0:
        return
    _TRACKING_KAFKA_E2E_LATENCY.observe(float(latency_ms) / 1000.0)


# Cardinalité bornée : on n'accepte qu'un petit jeu de raisons connues côté
# mobile, le reste retombe sur "_unknown" (évite l'explosion des labels par
# constraint_reason inventés côté client).
_KNOWN_CONSTRAINT_REASONS: frozenset[str] = frozenset(
    {
        "",
        "samsung_battery_optimized",
        "battery_optimized",
        "doze",
        "permission_revoked",
        "fg_permission_denied",
        "bg_permission_denied",
        "gps_provider_disabled",
        "fgs_killed",
        "low_fix_success_rate",
    }
)


def _norm_platform(platform: str | None) -> str:
    p = (platform or "").strip().lower()
    if p in ("ios", "android"):
        return p
    return "unknown"


def _norm_gps_provider(provider: str | None) -> str:
    p = (provider or "").strip().lower()
    if p in ("gps", "network", "passive", "fused", "cell", "wifi"):
        return p if p != "cell" else "network"
    return "unknown"


def inc_batch_rate_limited() -> None:
    if not _metrics_enabled() or _BATCH_RATE_LIMITED is None:
        return
    _BATCH_RATE_LIMITED.inc()


def inc_batch_points_received(*, location_mode: str, count: int = 1) -> None:
    if not _metrics_enabled() or _BATCH_POINTS_RECEIVED is None:
        return
    lm = _norm_mode(location_mode)
    _BATCH_POINTS_RECEIVED.labels(location_mode=lm).inc(count)


def inc_batch_points_canonical(*, location_mode: str) -> None:
    if not _metrics_enabled() or _BATCH_POINTS_CANONICAL is None:
        return
    _BATCH_POINTS_CANONICAL.labels(location_mode=_norm_mode(location_mode)).inc()


def inc_batch_points_observability(*, location_mode: str) -> None:
    if not _metrics_enabled() or _BATCH_POINTS_OBSERVABILITY is None:
        return
    _BATCH_POINTS_OBSERVABILITY.labels(location_mode=_norm_mode(location_mode)).inc()


def inc_batch_points_skipped(*, reason: str, location_mode: str) -> None:
    if not _metrics_enabled() or _BATCH_POINTS_SKIPPED is None:
        return
    r = reason if reason in ("dedup", "validation", "forbidden_mode", "location_service") else "_unknown"
    _BATCH_POINTS_SKIPPED.labels(reason=r, location_mode=_norm_mode(location_mode)).inc()


def inc_tracking_id_propagated(*, transport: str, propagated: bool) -> None:
    if not _metrics_enabled():
        return
    t = _norm_transport(transport)
    if propagated and _TRACKING_ID_PROPAGATED is not None:
        _TRACKING_ID_PROPAGATED.labels(transport=t).inc()
    elif not propagated and _TRACKING_ID_MISSING is not None:
        _TRACKING_ID_MISSING.labels(transport=t).inc()


def inc_canonical_redis_write(*, location_mode: str, transport: str) -> None:
    if not _metrics_enabled() or _CANONICAL_REDIS_WRITE is None:
        return
    t = _norm_transport(transport)
    _CANONICAL_REDIS_WRITE.labels(location_mode=_norm_mode(location_mode), transport=t).inc()


def inc_canonical_overwrite(*, outcome: str, location_mode: str) -> None:
    if not _metrics_enabled() or _CANONICAL_OVERWRITE is None:
        return
    oc = outcome if outcome in ("accepted", "rejected_older_than_canonical") else "_unknown"
    _CANONICAL_OVERWRITE.labels(outcome=oc, location_mode=_norm_mode(location_mode)).inc()


def observe_batch_latency_seconds(*, seconds: float) -> None:
    if not _metrics_enabled() or _BATCH_LATENCY is None:
        return
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return
    if s < 0 or s > 120.0:
        return
    _BATCH_LATENCY.observe(s)


def observe_gps_quality(
    *,
    platform: str | None,
    location_mode: str,
    transport: str,
    accuracy: float | None = None,
    speed: float | None = None,
    heading: float | None = None,
    provider: str | None = None,
) -> None:
    if not _metrics_enabled():
        return
    plat = _norm_platform(platform)
    lm = _norm_mode(location_mode)
    t = _norm_transport(transport)
    if accuracy is not None and _GPS_ACCURACY is not None:
        try:
            acc = float(accuracy)
            if 0 < acc <= 500:
                _GPS_ACCURACY.labels(platform=plat, location_mode=lm, transport=t).observe(acc)
        except (TypeError, ValueError):
            pass
    if speed is not None and _GPS_SPEED is not None:
        try:
            spd = float(speed)
            # Expo speed often m/s — convert if plausible m/s range
            if 0 <= abs(spd) <= 60:
                spd_kmh = abs(spd) * 3.6
            else:
                spd_kmh = abs(spd)
            if spd_kmh <= 200:
                _GPS_SPEED.labels(platform=plat, location_mode=lm, transport=t).observe(spd_kmh)
        except (TypeError, ValueError):
            pass
    if heading is not None and _GPS_HEADING is not None:
        try:
            hdg = float(heading)
            if 0 <= hdg <= 360:
                _GPS_HEADING.labels(platform=plat, location_mode=lm, transport=t).observe(hdg)
        except (TypeError, ValueError):
            pass
    if provider is not None and _GPS_PROVIDER is not None:
        _GPS_PROVIDER.labels(provider=_norm_gps_provider(provider), platform=plat).inc()


def inc_driver_device_health_received(*, constraint_reason: str | None) -> None:
    """Incrémente le compteur des heartbeats device-status reçus."""
    if not _metrics_enabled() or _DRIVER_DEVICE_HEALTH_RECEIVED is None:
        return
    raw = (constraint_reason or "").strip()
    cr = raw if raw in _KNOWN_CONSTRAINT_REASONS else "_unknown"
    _DRIVER_DEVICE_HEALTH_RECEIVED.labels(constraint_reason=cr).inc()


def inc_tracking_mission_live_missing_mission_id(
    *,
    transport: str,
    action: str = "downgraded",
) -> None:
    """Incrémente le compteur P0-C quand le mobile envoie mission_live sans mission_id."""
    if not _metrics_enabled() or _TRACKING_MISSION_LIVE_MISSING_MISSION_ID is None:
        return
    t = _norm_transport(transport)
    act = action if action in ("downgraded", "rejected") else "_unknown"
    _TRACKING_MISSION_LIVE_MISSING_MISSION_ID.labels(transport=t, action=act).inc()


def inc_tracking_invariant_violation(
    *,
    invariant_id: str,
    company_id: int | str | None = None,
    driver_id: int | None = None,
) -> None:
    """Compteur runtime violation invariant (INV-*). driver_id en log uniquement."""
    if not _metrics_enabled() or _TRACKING_INVARIANT_VIOLATION is None:
        return
    inv = (invariant_id or "UNKNOWN").strip().upper()
    if not inv.startswith("INV-"):
        inv = f"INV-{inv}"
    cid = str(company_id) if company_id is not None else "unknown"
    _TRACKING_INVARIANT_VIOLATION.labels(invariant_id=inv, company_id=cid).inc()
    _ = driver_id


def observe_tracking_position_freshness_seconds(
    *,
    freshness_seconds: float,
    company_id: int | str | None = None,
    location_mode: str | None = None,
) -> None:
    """Histogramme fraîcheur position (N3 SLO)."""
    if not _metrics_enabled() or _TRACKING_POSITION_FRESHNESS is None:
        return
    try:
        val = max(0.0, float(freshness_seconds))
    except (TypeError, ValueError):
        return
    cid = str(company_id) if company_id is not None else "unknown"
    _TRACKING_POSITION_FRESHNESS.labels(
        company_id=cid,
        location_mode=_norm_mode(location_mode),
    ).observe(val)


_KAFKA_PERSIST_STATUSES = frozenset(
    {
        "accepted_canonical",
        "accepted_observability_only",
        "skipped",
        "failed",
    }
)


def inc_tracking_http_accepted_async(*, location_mode: str) -> None:
    """Compteur HTTP 202 — position acceptée pour enqueue Kafka."""
    if not _metrics_enabled() or _TRACKING_HTTP_ACCEPTED_ASYNC is None:
        return
    _TRACKING_HTTP_ACCEPTED_ASYNC.labels(location_mode=_norm_mode(location_mode)).inc()


def inc_tracking_kafka_persist(*, accept_status: str) -> None:
    """Compteur persist consumer — labels finis uniquement (pas driver_id/company_id)."""
    if not _metrics_enabled() or _TRACKING_KAFKA_PERSIST is None:
        return
    st = accept_status if accept_status in _KAFKA_PERSIST_STATUSES else "failed"
    _TRACKING_KAFKA_PERSIST.labels(accept_status=st).inc()


_TRACKING_INVALID_CONFIG_REASONS = frozenset({"async_without_persist"})


def inc_tracking_invalid_config(*, reason: str) -> None:
    """Compteur config invalide au démarrage ingest_consumer (labels finis)."""
    if not _metrics_enabled() or _TRACKING_INVALID_CONFIG is None:
        return
    r = reason if reason in _TRACKING_INVALID_CONFIG_REASONS else "async_without_persist"
    _TRACKING_INVALID_CONFIG.labels(reason=r).inc()
