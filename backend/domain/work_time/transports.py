"""Décision unique : une course compte-t-elle comme un transport."""

from __future__ import annotations

from datetime import datetime

_COMPLETED = frozenset({"COMPLETED", "RETURN_COMPLETED"})
_CANCELLED = frozenset({"CANCELED", "CANCELLED"})
# États d'assignation qui prouvent que le chauffeur est arrivé sur place,
# ou un jalon ultérieur fiable. CANCELLED n'en fait pas partie : c'est
# l'annulation elle-même, pas une preuve d'arrivée.
_ON_SITE = frozenset(
    {
        "ARRIVED_PICKUP",
        "ONBOARD",
        "EN_ROUTE_DROPOFF",
        "ARRIVED_DROPOFF",
        "COMPLETED",
    }
)


def _token(value: object) -> str:
    raw = str(getattr(value, "value", value) or "").strip().upper()
    if "." in raw:
        raw = raw.rsplit(".", 1)[-1]
    return raw


def driver_arrived_on_site(
    *,
    arrived_at: datetime | None,
    assignment_status: object | None = None,
) -> bool:
    """Preuve d'arrivée : horodatage, ou assignation ARRIVED_PICKUP ou après."""
    if arrived_at is not None:
        return True
    return _token(assignment_status) in _ON_SITE


def counts_as_transport(booking) -> bool:
    """1 transport si la course est terminée, ou annulée après arrivée sur place.

    ``booking`` expose ``status`` (ou ``status_key``), ``arrived_at`` et,
    le cas échéant, ``assignment_status``.
    """
    status = _token(
        getattr(booking, "status_key", None) or getattr(booking, "status", None)
    )
    if status in _COMPLETED:
        return True
    if status in _CANCELLED:
        return driver_arrived_on_site(
            arrived_at=getattr(booking, "arrived_at", None),
            assignment_status=getattr(booking, "assignment_status", None),
        )
    return False
