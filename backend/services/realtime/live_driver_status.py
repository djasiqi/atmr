"""Statut mission / affichage chauffeur pour fanout temps réel (Socket + HTTP).

Même logique que l'historique dans sockets/chat.py — centralisée pour éviter
« busy » dès qu'un mission_id est présent alors que la course est seulement ASSIGNED.

P1 : ``resolve_active_booking_id_for_driver`` / ``sanitize_fanout_mission_id``
passent par ``authoritative_tracking_mission`` (SINGLE uniquement, pas de pick
silencieux par ``updated_at``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from models import Booking
from models.enums import BookingStatus

# Fenêtre ASSIGNED alignée sur stale_fix_watchdog (évite import circulaire).
_ASSIGNED_LEAD_BEFORE_SEC = int(
    os.getenv("STALE_FIX_WATCHDOG_ASSIGNED_LEAD_SEC", "1800")
)
_ASSIGNED_GRACE_AFTER_SEC = int(
    os.getenv("STALE_FIX_WATCHDOG_ASSIGNED_GRACE_SEC", "3600")
)


class TrackingMissionResolutionState(str, Enum):
    NONE = "NONE"
    SINGLE = "SINGLE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class TrackingMissionResolution:
    """Résolution authoritative de la mission trackable d'un chauffeur (P0-B)."""

    state: TrackingMissionResolutionState
    mission_id: int | None
    status: str | None
    trackable_now: bool
    reason: str
    candidate_ids: tuple[int, ...]


def _parse_dt(value) -> datetime | None:
    """Normalise un datetime métier en UTC aware.

    Les ``DateTime(timezone=False)`` Postgres (ex. ``Booking.scheduled_time``)
    sont naïfs **Europe/Zurich**, pas UTC — utiliser ``to_utc_from_db``.
    """
    if value is None or value == "":
        return None
    from shared.time_utils import to_utc_from_db

    return to_utc_from_db(value)


def assigned_in_tracking_window(
    scheduled_at,
    time_confirmed,
    now: datetime | None = None,
) -> bool:
    """ASSIGNED confirmé dont l'heure prévue est dans [T−lead, T+grace]."""
    if not time_confirmed:
        return False
    dt = _parse_dt(scheduled_at)
    if dt is None:
        return False
    ref = now or datetime.now(UTC)
    return (
        (dt.timestamp() - _ASSIGNED_LEAD_BEFORE_SEC)
        <= ref.timestamp()
        <= (dt.timestamp() + _ASSIGNED_GRACE_AFTER_SEC)
    )


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
    """ID de la mission trackable authoritative, ou None.

    P1 : délègue à ``authoritative_tracking_mission`` — retourne ``mission_id``
    seulement si état ``SINGLE`` (pas de sélection silencieuse par ``updated_at``).
    """
    resolution = authoritative_tracking_mission(driver_id)
    if resolution.state == TrackingMissionResolutionState.SINGLE:
        return resolution.mission_id
    return None


def authoritative_tracking_mission(
    driver_id: int,
    *,
    now: datetime | None = None,
) -> TrackingMissionResolution:
    """Résout la mission de tracking authoritaire (NONE | SINGLE | AMBIGUOUS).

    Priorité déterministe :
    1. IN_PROGRESS (si >1 ⇒ AMBIGUOUS)
    2. EN_ROUTE (si >1 ⇒ AMBIGUOUS)
    3. ASSIGNED dans la fenêtre tracking (si >1 ⇒ AMBIGUOUS)
    4. sinon NONE

    ``IN_PROGRESS`` + ``ASSIGNED`` ⇒ SINGLE sur ``IN_PROGRESS`` (pas AMBIGUOUS).
    """
    ref = now or datetime.now(UTC)
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
        .with_entities(
            Booking.id,
            Booking.status,
            Booking.scheduled_time,
            Booking.time_confirmed,
        )
        .all()
    )

    by_status: dict[str, list[tuple[int, object, object]]] = {
        BookingStatus.IN_PROGRESS.value: [],
        BookingStatus.EN_ROUTE.value: [],
        BookingStatus.ASSIGNED.value: [],
    }
    for row in rows:
        bid = getattr(row, "id", None)
        if bid is None:
            continue
        try:
            mission_id = int(bid)
        except (TypeError, ValueError):
            continue
        status = getattr(row, "status", None)
        status_value = str(getattr(status, "value", status) or "").upper()
        if status_value not in by_status:
            continue
        by_status[status_value].append(
            (
                mission_id,
                getattr(row, "scheduled_time", None),
                getattr(row, "time_confirmed", None),
            )
        )

    for status_key in (
        BookingStatus.IN_PROGRESS.value,
        BookingStatus.EN_ROUTE.value,
    ):
        candidates = by_status[status_key]
        if len(candidates) > 1:
            ids = tuple(sorted(c[0] for c in candidates))
            return TrackingMissionResolution(
                state=TrackingMissionResolutionState.AMBIGUOUS,
                mission_id=None,
                status=status_key,
                trackable_now=False,
                reason=f"ambiguous_{status_key.lower()}",
                candidate_ids=ids,
            )
        if len(candidates) == 1:
            mid = candidates[0][0]
            return TrackingMissionResolution(
                state=TrackingMissionResolutionState.SINGLE,
                mission_id=mid,
                status=status_key,
                trackable_now=True,
                reason="single_live_mission",
                candidate_ids=(mid,),
            )

    assigned_trackable: list[int] = []
    assigned_all: list[int] = []
    for mid, scheduled, confirmed in by_status[BookingStatus.ASSIGNED.value]:
        assigned_all.append(mid)
        if assigned_in_tracking_window(scheduled, confirmed, ref):
            assigned_trackable.append(mid)

    if len(assigned_trackable) > 1:
        ids = tuple(sorted(assigned_trackable))
        return TrackingMissionResolution(
            state=TrackingMissionResolutionState.AMBIGUOUS,
            mission_id=None,
            status=BookingStatus.ASSIGNED.value,
            trackable_now=False,
            reason="ambiguous_assigned_in_window",
            candidate_ids=ids,
        )
    if len(assigned_trackable) == 1:
        mid = assigned_trackable[0]
        return TrackingMissionResolution(
            state=TrackingMissionResolutionState.SINGLE,
            mission_id=mid,
            status=BookingStatus.ASSIGNED.value,
            trackable_now=True,
            reason="single_assigned_in_window",
            candidate_ids=(mid,),
        )

    if assigned_all:
        return TrackingMissionResolution(
            state=TrackingMissionResolutionState.NONE,
            mission_id=None,
            status=BookingStatus.ASSIGNED.value,
            trackable_now=False,
            reason="assigned_outside_tracking_window",
            candidate_ids=tuple(sorted(assigned_all)),
        )

    return TrackingMissionResolution(
        state=TrackingMissionResolutionState.NONE,
        mission_id=None,
        status=None,
        trackable_now=False,
        reason="no_active_booking",
        candidate_ids=(),
    )


def sanitize_fanout_mission_id(
    driver_id: int,
    client_mission_id: int | None,  # noqa: ARG001 — conservé pour signature hot path
) -> int | None:
    """Ne fanoute un ``mission_id`` que si la résolution authoritative est SINGLE.

    P1 : ``resolution.mission_id`` si SINGLE, sinon None (AMBIGUOUS / NONE).
    Le ``client_mission_id`` n'est plus utilisé pour choisir une mission.
    """
    resolution = authoritative_tracking_mission(driver_id)
    if resolution.state != TrackingMissionResolutionState.SINGLE:
        return None
    return resolution.mission_id


def resolve_driver_status_for_fanout(
    *,
    mission_status: str,
    is_active: bool,
    presence_status: str,
    is_available: bool = True,
) -> str:
    """Statut affiché entreprise : offline | off_duty | busy | assigned | available | *_constrained.

    SoT métier en-service : ``Driver.is_available`` (pas le socket, pas la fraîcheur GPS).
    ``is_available=False`` → ``off_duty`` (hors service), jamais ``available``.
    """
    if not is_active:
        return "offline"
    if not is_available:
        return "off_duty"
    if mission_status in {
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    }:
        return "busy"
    if mission_status == BookingStatus.ASSIGNED.value:
        if presence_status == "degraded_constrained":
            return "assigned_constrained"
        return "assigned"
    if presence_status == "offline":
        return "offline"
    if presence_status == "degraded_constrained":
        return "available_constrained"
    return "available"
