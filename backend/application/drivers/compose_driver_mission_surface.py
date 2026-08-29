"""Composition surface chauffeur (ARRIVED-SOT-2).

BookingStatus reste le cycle réservation (pas d'ARRIVED).
AssignmentStatus porte l'état opérationnel.
La surface GET chauffeur expose une vue composée durable.
"""

from __future__ import annotations

from typing import Any

from models.enums import AssignmentStatus, BookingStatus

# Statut composé pour la surface chauffeur uniquement (pas BookingStatus).
COMPOSED_DRIVER_STATUS_ARRIVED = "arrived"
MISSION_MILESTONE_ARRIVED = "ARRIVED"


def _status_token(value: Any) -> str:
    if value is None:
        return ""
    raw = getattr(value, "value", value)
    return str(raw).strip().upper()


def is_booking_en_route(booking_status: Any) -> bool:
    return _status_token(booking_status) == BookingStatus.EN_ROUTE.value


def is_assignment_arrived_pickup(assignment_status: Any) -> bool:
    return _status_token(assignment_status) == AssignmentStatus.ARRIVED_PICKUP.value


def should_compose_arrived(
    *, booking_status: Any, assignment_status: Any | None
) -> bool:
    """EN_ROUTE + ARRIVED_PICKUP ⇒ surface ARRIVED durable."""
    if assignment_status is None:
        return False
    return is_booking_en_route(booking_status) and is_assignment_arrived_pickup(
        assignment_status
    )


def apply_arrived_composition(payload: dict[str, Any]) -> dict[str, Any]:
    """Applique les champs composés sur un dict déjà sérialisé (mutatif + return)."""
    payload["mission_milestone"] = MISSION_MILESTONE_ARRIVED
    # Aligné sur Booking.serialize (status en minuscules) pour resolveDriverStatusForUx.
    payload["status"] = COMPOSED_DRIVER_STATUS_ARRIVED
    return payload


def compose_driver_mission_payload(
    payload: dict[str, Any],
    *,
    assignment_status: Any | None,
) -> dict[str, Any]:
    """Compose la surface chauffeur à partir du payload booking + Assignment."""
    out = dict(payload)
    booking_status = out.get("status")
    if should_compose_arrived(
        booking_status=booking_status, assignment_status=assignment_status
    ):
        return apply_arrived_composition(out)
    return out


def latest_assignment_by_booking_id(
    assignments: list[Any],
) -> dict[int, Any]:
    """Une entrée par booking_id : assignment courant (resolver unique P0-B)."""
    from services.dispatch.assignment_resolver import pick_current_assignment

    grouped: dict[int, list[Any]] = {}
    for a in assignments:
        bid = getattr(a, "booking_id", None)
        if bid is None:
            continue
        grouped.setdefault(int(bid), []).append(a)
    return {
        bid: pick_current_assignment(items) for bid, items in grouped.items()
    }


def latest_assignment_status_by_booking_id(
    assignments: list[Any],
) -> dict[int, Any]:
    """Une entrée par booking_id : statut de l'assignment courant."""
    return {
        bid: getattr(a, "status", None)
        for bid, a in latest_assignment_by_booking_id(assignments).items()
    }


def attach_assignment_identity(
    payload: dict[str, Any], assignment: Any | None
) -> dict[str, Any]:
    """Attache l'identité de lifecycle (P1) : assignment_id + mission_revision.

    Le mobile s'en sert pour ignorer tout snapshot plus ancien que son état
    local (anti-régression), et pour détecter un changement de lifecycle.
    """
    if assignment is None:
        payload.setdefault("assignment_id", None)
        payload.setdefault("mission_revision", None)
        return payload
    payload["assignment_id"] = getattr(assignment, "id", None)
    try:
        payload["mission_revision"] = int(
            getattr(assignment, "revision", 0) or 0
        )
    except (TypeError, ValueError):
        payload["mission_revision"] = 0
    return payload
