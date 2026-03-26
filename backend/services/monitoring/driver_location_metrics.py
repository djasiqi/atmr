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


def _norm_mode(mode: str | None) -> str:
    """Aligné sur ``normalize_location_mode`` (valeurs invalides → ``mission_live``)."""
    m = (mode or "").strip()
    if m in ("mission_live", "availability_presence", "passive_last_known"):
        return m
    return "mission_live"


try:
    from prometheus_client import Counter
except ImportError:
    Counter = None

_RECEIVED = None
_PROCESSED = None
_FANOUT = None

if Counter is not None:
    _RECEIVED = Counter(
        "driver_location_received_total",
        "Positions reçues (HTTP ou socket), avant traitement complet",
        ["transport", "location_mode"],
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


def inc_received(*, transport: str, location_mode: str) -> None:
    if not _metrics_enabled() or _RECEIVED is None:
        return
    lm = _norm_mode(location_mode)
    t = transport if transport in ("http", "socket", "socket_batch") else "http"
    _RECEIVED.labels(transport=t, location_mode=lm).inc()


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
