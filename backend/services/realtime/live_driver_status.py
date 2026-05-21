"""Statut mission / affichage chauffeur pour fanout temps réel (Socket + HTTP).

Même logique que l'historique dans sockets/chat.py — centralisée pour éviter
« busy » dès qu'un mission_id est présent alors que la course est seulement ASSIGNED.
"""

from __future__ import annotations

from models import Booking
from models.enums import BookingStatus


def resolve_mission_status_for_driver(driver_id: int) -> str:
    """Statut métier agrégé pour les réservations actives du chauffeur (priorité IN_PROGRESS > EN_ROUTE > ASSIGNED)."""
    statuses = (
        BookingStatus.ASSIGNED.value,
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    )
    rows = (
        Booking.query.filter(
            Booking.driver_id == driver_id,
            Booking.status.in_(statuses),
        )
        .with_entities(Booking.status)
        .all()
    )
    found: set[str] = set()
    for row in rows:
        raw = getattr(row, "status", None)
        status_value = getattr(raw, "value", raw)
        found.add(str(status_value or "").upper())
    if BookingStatus.IN_PROGRESS.value in found:
        return BookingStatus.IN_PROGRESS.value
    if BookingStatus.EN_ROUTE.value in found:
        return BookingStatus.EN_ROUTE.value
    if BookingStatus.ASSIGNED.value in found:
        return BookingStatus.ASSIGNED.value
    return "NONE"


def resolve_active_booking_id_for_driver(driver_id: int) -> int | None:
    """ID de la course active la plus récente (ASSIGNED / EN_ROUTE / IN_PROGRESS), ou None."""
    statuses = (
        BookingStatus.ASSIGNED.value,
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    )
    row = (
        Booking.query.filter(
            Booking.driver_id == driver_id,
            Booking.status.in_(statuses),
        )
        .with_entities(Booking.id)
        .order_by(Booking.updated_at.desc())
        .first()
    )
    if row is None:
        return None
    bid = getattr(row, "id", None)
    if bid is None:
        return None
    try:
        return int(bid)
    except (TypeError, ValueError):
        return None


def sanitize_fanout_mission_id(
    driver_id: int,
    client_mission_id: int | None,
) -> int | None:
    """Ne fanoute pas un ``mission_id`` GPS obsolète après fin de course."""
    active_id = resolve_active_booking_id_for_driver(driver_id)
    if active_id is None:
        return None
    if client_mission_id is None:
        return active_id
    try:
        client = int(client_mission_id)
    except (TypeError, ValueError):
        return active_id
    return active_id


def resolve_driver_status_for_fanout(
    *,
    mission_status: str,
    is_active: bool,
    presence_status: str,
) -> str:
    """Statut affiché entreprise : offline | busy | assigned | available."""
    if not is_active:
        return "offline"
    if mission_status in {
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    }:
        return "busy"
    if mission_status == BookingStatus.ASSIGNED.value:
        return "assigned"
    if presence_status == "offline":
        return "offline"
    return "available"
