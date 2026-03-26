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


def inc_socketio_event(event_name: str) -> None:
    if SOCKETIO_EVENTS_TOTAL is not None:
        SOCKETIO_EVENTS_TOTAL.labels(event=event_name).inc()


def inc_socketio_alias_emit(alias: str) -> None:
    """Compte les alias lirie.* (observabilité : volume alias vs primary)."""
    if SOCKETIO_ALIAS_EMITS_TOTAL is not None:
        SOCKETIO_ALIAS_EMITS_TOTAL.labels(alias=alias).inc()
