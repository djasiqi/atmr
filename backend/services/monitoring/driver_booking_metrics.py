"""Métriques Prometheus — mutations réservation chauffeur (403 métier, etc.)."""

from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram
except ImportError:  # pragma: no cover
    Counter = None  # type: ignore[misc, assignment]
    Histogram = None  # type: ignore[misc, assignment]

# Aligné mobile missionSyncTypes + valeur agrégée
_BOOKINGS_SINCE_TRIGGERS: frozenset[str] = frozenset(
    {
        "unknown",
        "socket_connect",
        "foreground",
        "degraded_interval",
        "reconcile_now",
        "reconcile_active",
        "manual_screen",
        "hydrate_empty",
        "socket_booking_event",
    }
)

_BOOKINGS_SINCE_REQUESTS: "Counter | None" = None
_BOOKINGS_SINCE_DURATION: "Histogram | None" = None


def _get_bookings_since_metrics() -> tuple["Counter | None", "Histogram | None"]:
    global _BOOKINGS_SINCE_REQUESTS, _BOOKINGS_SINCE_DURATION
    if Counter is None or Histogram is None:
        return None, None
    if _BOOKINGS_SINCE_REQUESTS is None:
        _BOOKINGS_SINCE_REQUESTS = Counter(
            "driver_bookings_since_requests_total",
            "GET /driver/me/bookings/since (label trigger_reason = X-LIRIE-Sync-Trigger normalisé)",
            ["trigger_reason"],
        )
    if _BOOKINGS_SINCE_DURATION is None:
        _BOOKINGS_SINCE_DURATION = Histogram(
            "driver_bookings_since_duration_seconds",
            "Durée handler GET /driver/me/bookings/since",
            ["trigger_reason"],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
        )
    return _BOOKINGS_SINCE_REQUESTS, _BOOKINGS_SINCE_DURATION


def normalize_bookings_since_trigger(header_val: str | None) -> str:
    """Header X-LIRIE-Sync-Trigger : enum fermée, sinon unknown."""
    if not header_val or not str(header_val).strip():
        return "unknown"
    t = str(header_val).strip()
    if t in _BOOKINGS_SINCE_TRIGGERS:
        return t
    return "unknown"


def observe_driver_bookings_since_request(*, trigger_reason: str, duration_seconds: float) -> None:
    """Compteur + histogramme pour une requête GET /me/bookings/since terminée."""
    tr = trigger_reason if trigger_reason in _BOOKINGS_SINCE_TRIGGERS else "unknown"
    c, h = _get_bookings_since_metrics()
    if c is not None:
        c.labels(trigger_reason=tr).inc()
    if h is not None:
        h.labels(trigger_reason=tr).observe(max(0.0, float(duration_seconds)))

_DRIVER_BOOKING_STATUS_FORBIDDEN: "Counter | None" = None


def _get_forbidden_counter() -> "Counter | None":
    global _DRIVER_BOOKING_STATUS_FORBIDDEN
    if Counter is None:
        return None
    if _DRIVER_BOOKING_STATUS_FORBIDDEN is None:
        _DRIVER_BOOKING_STATUS_FORBIDDEN = Counter(
            "driver_booking_status_forbidden_total",
            "Driver booking status PUT forbidden (métier)",
            ["code"],
        )
    return _DRIVER_BOOKING_STATUS_FORBIDDEN


def inc_driver_booking_status_forbidden(code: str) -> None:
    """Incrémente le compteur 403 métier sur PUT /driver/me/bookings/.../status."""
    c = _get_forbidden_counter()
    if c is None:
        return
    c.labels(code=code).inc()


_REASSIGN_FANOUT: "Counter | None" = None


def _get_reassign_fanout_counter() -> "Counter | None":
    global _REASSIGN_FANOUT
    if Counter is None:
        return None
    if _REASSIGN_FANOUT is None:
        _REASSIGN_FANOUT = Counter(
            "driver_booking_reassigned_fanout_total",
            "booking_reassigned émis vers l'ancien chauffeur (fanout)",
        )
    return _REASSIGN_FANOUT


def inc_booking_reassigned_fanout() -> None:
    c = _get_reassign_fanout_counter()
    if c is None:
        return
    c.inc()
