"""Métriques Prometheus — mutations réservation chauffeur (403 métier, etc.)."""

from __future__ import annotations

try:
    from prometheus_client import Counter
except ImportError:  # pragma: no cover
    Counter = None  # type: ignore[misc, assignment]

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
