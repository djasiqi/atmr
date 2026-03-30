"""Métriques Prometheus LIRIE (Socket.IO, payloads) — import optionnel prometheus_client."""

from __future__ import annotations

import os

try:
    from prometheus_client import (
        Counter,
        Histogram,
    )

    _PROM = True
except ImportError:
    _PROM = False
    Counter = None
    Histogram = None

if _PROM and Counter is not None:
    DELAY_LIVE_INVALIDATE_EMITTED_TOTAL = Counter(
        "delay_live_invalidate_emitted_total",
        "Émissions delay_live_invalidate (après throttle P3). Corréler avec Traefik sur GET /company_dispatch/delays/live.",
    )
    DELAY_LIVE_INVALIDATE_SKIPPED_TOTAL = Counter(
        "delay_live_invalidate_skipped_total",
        "Invalidations live delays ignorées (ex. throttle P3). Ratio emitted/skipped utile avec le volume HTTP ci-dessus.",
        ["reason"],
    )
    SOCKETIO_EVENTS_TOTAL = Counter(
        "socketio_events_total",
        "Événements Socket.IO émis via _safe_emit",
        ["event"],
    )
    SOCKETIO_ALIAS_EMITS_TOTAL = Counter(
        "socketio_alias_emits_total",
        "Émissions d’alias lirie.* (hors comptage principal socketio_events_total ; canon = événement primary)",
        ["alias"],
    )
else:
    DELAY_LIVE_INVALIDATE_EMITTED_TOTAL = None
    DELAY_LIVE_INVALIDATE_SKIPPED_TOTAL = None
    SOCKETIO_EVENTS_TOTAL = None
    SOCKETIO_ALIAS_EMITS_TOTAL = None

if _PROM and Histogram is not None:
    COMPANY_RESERVATIONS_RESPONSE_BYTES = Histogram(
        "company_reservations_response_bytes",
        "Taille approximative du JSON de GET /companies/me/reservations (liste)",
        buckets=[
            1024,
            4096,
            16384,
            65536,
            262144,
            1048576,
            4194304,
            1.6777216e7,
        ],
    )
else:
    COMPANY_RESERVATIONS_RESPONSE_BYTES = None


def observe_reservations_payload_size(num_bytes: int) -> None:
    """Enregistre la taille du payload si METRICS_RESERVATIONS_PAYLOAD=true."""
    if os.getenv("METRICS_RESERVATIONS_PAYLOAD", "false").lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    if COMPANY_RESERVATIONS_RESPONSE_BYTES is not None:
        COMPANY_RESERVATIONS_RESPONSE_BYTES.observe(max(0, float(num_bytes)))


def inc_delay_live_invalidate_emitted() -> None:
    if DELAY_LIVE_INVALIDATE_EMITTED_TOTAL is not None:
        DELAY_LIVE_INVALIDATE_EMITTED_TOTAL.inc()


def inc_delay_live_invalidate_skipped(reason: str) -> None:
    if DELAY_LIVE_INVALIDATE_SKIPPED_TOTAL is not None:
        DELAY_LIVE_INVALIDATE_SKIPPED_TOTAL.labels(reason=reason).inc()


def inc_socketio_event(event_name: str) -> None:
    if SOCKETIO_EVENTS_TOTAL is not None:
        SOCKETIO_EVENTS_TOTAL.labels(event=event_name).inc()


def inc_socketio_alias_emit(alias: str) -> None:
    """Compte les alias lirie.* (observabilité : volume alias vs primary)."""
    if SOCKETIO_ALIAS_EMITS_TOTAL is not None:
        SOCKETIO_ALIAS_EMITS_TOTAL.labels(alias=alias).inc()
