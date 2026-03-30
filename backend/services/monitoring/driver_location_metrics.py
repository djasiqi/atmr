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
    from prometheus_client import Counter, Histogram
except ImportError:
    Counter = None
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
else:
    _BATCH_INGEST_SIZE = None


def inc_dedup_skipped(
    *,
    reason: str,
    location_mode: str,
    transport: str,
) -> None:
    if not _metrics_enabled() or _DEDUP_SKIPPED is None:
        return
    lm = _norm_mode(location_mode)
    t = transport if transport in ("http", "socket", "socket_batch") else "http"
    r = reason if reason in ("duplicate_event_id", "duplicate_proximity") else "_unknown"
    _DEDUP_SKIPPED.labels(reason=r, location_mode=lm, transport=t).inc()


def inc_received(*, transport: str, location_mode: str) -> None:
    """Incrémente received + ingested (mêmes labels) pour les points non skippés en amont."""
    if not _metrics_enabled() or _RECEIVED is None:
        return
    lm = _norm_mode(location_mode)
    t = transport if transport in ("http", "socket", "socket_batch") else "http"
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
    t = transport if transport in ("http", "socket", "socket_batch") else "http"
    st = accept_status if accept_status in (
        "accepted_canonical",
        "accepted_observability_only",
        "rejected_invalid",
        "skipped",
    ) else "_unknown"
    _PROCESSED.labels(
        accept_status=st,
        accept_reason=ar,
        location_mode=lm,
        transport=t,
    ).inc()


def inc_fanout(*, event: str, accept_status: str) -> None:
    if not _metrics_enabled() or _FANOUT is None:
        return
    ev = event if event in ("driver_location_update", "driver_live_state_update") else "_unknown"
    st = accept_status if accept_status in (
        "accepted_canonical",
        "accepted_observability_only",
    ) else "_unknown"
    _FANOUT.labels(event=ev, accept_status=st).inc()


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
