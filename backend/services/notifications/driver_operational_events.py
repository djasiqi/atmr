"""Événements opérationnels chauffeur : transitions métier, texte sobre, id stable.

Le push n'est pas la source de vérité. Le mobile refetch la course au clic.
Aucun nom de patient, pathologie, médecin ou note médicale dans le texte système.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

EVENT_DRIVER_ASSIGNED = "DRIVER_ASSIGNED"
EVENT_DRIVER_UNASSIGNED = "DRIVER_UNASSIGNED"
EVENT_ROUTE_CHANGED = "ROUTE_CHANGED"
EVENT_SCHEDULE_CHANGED = "SCHEDULE_CHANGED"
EVENT_BOOKING_CANCELLED = "BOOKING_CANCELLED"
EVENT_BOOKING_CHANGED = "BOOKING_CHANGED"

REASON_REASSIGNED = "reassigned"
REASON_UNASSIGNED = "unassigned"

ROUTE_CHANGE_FIELDS = frozenset(
    {
        "pickup_location",
        "dropoff_location",
        "return_location",
        "return_address",
        "stops",
        "waypoints",
        "route_steps",
        "intermediate_stops",
        "legs",
    }
)
SCHEDULE_CHANGE_FIELDS = frozenset(
    {
        "scheduled_time",
        "pickup_time",
        "appointment_time",
        "arrival_time",
        "return_time",
        "mission_date",
        "scheduled_date",
    }
)

_LEGACY_TYPE = {
    EVENT_DRIVER_ASSIGNED: "booking_assigned",
    EVENT_DRIVER_UNASSIGNED: "booking_reassigned",
    EVENT_ROUTE_CHANGED: "booking_updated",
    EVENT_SCHEDULE_CHANGED: "booking_updated",
    EVENT_BOOKING_CHANGED: "booking_updated",
    EVENT_BOOKING_CANCELLED: "booking_cancelled",
}

_SENSITIVE_KEYS = frozenset(
    {
        "client_name",
        "client_display_name",
        "client",
        "patient_name",
        "notes",
        "notes_medical",
        "doctor_name",
        "medical_facility",
        "hospital_service",
        "pickup_access_notes",
        "dropoff_access_notes",
    }
)


@dataclass(frozen=True)
class DriverOperationalNotification:
    """Un push chauffeur pour une transition métier."""

    event_id: str
    event_type: str
    target_driver_id: int
    booking_id: int
    title: str
    body: str
    payload: dict[str, Any]


def stable_event_id(
    *,
    mutation_id: str,
    event_type: str,
    driver_id: int,
    booking_id: int,
) -> str:
    """Même mutation + même destinataire = même event_id, y compris au retry."""
    raw = f"lirie:driver-operational:{mutation_id}:{event_type}:{driver_id}:{booking_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def classify_change_kinds(changes: dict[str, Any] | None) -> set[str]:
    """Retourne {'route', 'schedule'} selon les champs réellement opérationnels."""
    if not isinstance(changes, dict):
        return set()
    keys = set(changes.keys())
    kinds: set[str] = set()
    if keys.intersection(ROUTE_CHANGE_FIELDS):
        kinds.add("route")
    if keys.intersection(SCHEDULE_CHANGE_FIELDS):
        kinds.add("schedule")
    return kinds


def _hhmm(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if "T" in text and len(text) >= 16:
        return text.replace("Z", "")[11:16]
    if len(text) >= 5 and text[2:3] == ":":
        return text[:5]
    return ""


def _change_endpoint(changes: dict[str, Any] | None, field: str, side: str) -> str:
    if not isinstance(changes, dict):
        return ""
    raw = changes.get(field)
    if isinstance(raw, dict):
        return _hhmm(raw.get(side) or raw.get("to" if side == "to" else "from"))
    if side == "to":
        return _hhmm(raw)
    return ""


def time_label_from_context(context: dict[str, Any] | None) -> str:
    """Heure HH:MM déjà connue, sans reconstruire le métier côté mobile."""
    if not isinstance(context, dict):
        return ""
    formatted = context.get("time_formatted") or context.get("time_formatted_local")
    if formatted:
        label = _hhmm(formatted)
        if label:
            return label
        text = str(formatted).strip()
        return text[:16]
    return _hhmm(context.get("scheduled_time"))


def render_operational_copy(
    event_type: str,
    *,
    time_label: str | None = None,
    old_time_label: str | None = None,
    new_time_label: str | None = None,
    unassign_reason: str | None = None,
) -> tuple[str, str]:
    """Titre et corps sobres. Jamais de donnée médicale."""
    when = (time_label or "").strip()
    old_when = (old_time_label or "").strip()
    new_when = (new_time_label or "").strip()

    if event_type == EVENT_DRIVER_ASSIGNED:
        body = (
            f"Un transport vous a été assigné pour {when}."
            if when
            else "Un transport vous a été assigné."
        )
        return "Nouveau transport assigné", body

    if event_type == EVENT_SCHEDULE_CHANGED:
        if old_when and new_when:
            body = f"Votre transport prévu à {old_when} est désormais prévu à {new_when}."
        elif new_when:
            body = f"Votre transport est désormais prévu à {new_when}."
        else:
            body = "L'horaire de votre transport a été modifié."
        return "Horaire modifié", body

    if event_type == EVENT_ROUTE_CHANGED:
        body = (
            f"Le trajet de votre transport de {when} a été modifié."
            if when
            else "Le trajet de votre transport a été modifié."
        )
        return "Itinéraire modifié", body

    if event_type == EVENT_BOOKING_CHANGED:
        return (
            "Transport modifié",
            "L'itinéraire et l'horaire de votre transport ont été modifiés.",
        )

    if event_type == EVENT_DRIVER_UNASSIGNED:
        if unassign_reason == REASON_REASSIGNED:
            return (
                "Transport réattribué",
                "Ce transport a été réattribué à un autre chauffeur.",
            )
        return "Transport retiré", "Ce transport ne vous est plus assigné."

    if event_type == EVENT_BOOKING_CANCELLED:
        body = (
            f"Le transport prévu à {when} a été annulé."
            if when
            else "Le transport a été annulé."
        )
        return "Transport annulé", body

    return "Transport modifié", "Votre transport a été modifié."


def build_operational_payload(
    *,
    event_id: str,
    event_type: str,
    booking_id: int,
    mission_anchor_booking_id: int | None,
    occurred_at: str,
    reason: str | None = None,
) -> dict[str, Any]:
    anchor = mission_anchor_booking_id or booking_id
    deep_link = f"lirie://driver/bookings/{booking_id}"
    payload: dict[str, Any] = {
        "event_id": event_id,
        "event_type": event_type,
        "type": _LEGACY_TYPE.get(event_type, "booking_updated"),
        "booking_id": booking_id,
        "mission_id": booking_id,
        "mission_anchor_booking_id": anchor,
        "occurred_at": occurred_at,
        "deep_link": deep_link,
        "deepLink": deep_link,
        "dedupe_key": f"event:{event_id}",
        "recipient_role": "driver",
        "channelId": "mission_updates",
    }
    if reason:
        payload["reason"] = reason
    return payload


def strip_sensitive_push_data(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in _SENSITIVE_KEYS}


def _notification(
    *,
    event_type: str,
    driver_id: int,
    booking_id: int,
    mutation_id: str,
    time_label: str | None,
    old_time_label: str | None,
    new_time_label: str | None,
    mission_anchor_booking_id: int | None,
    occurred_at: str,
    unassign_reason: str | None = None,
) -> DriverOperationalNotification:
    event_id = stable_event_id(
        mutation_id=mutation_id,
        event_type=event_type,
        driver_id=driver_id,
        booking_id=booking_id,
    )
    title, body = render_operational_copy(
        event_type,
        time_label=time_label,
        old_time_label=old_time_label,
        new_time_label=new_time_label,
        unassign_reason=unassign_reason,
    )
    return DriverOperationalNotification(
        event_id=event_id,
        event_type=event_type,
        target_driver_id=driver_id,
        booking_id=booking_id,
        title=title,
        body=body,
        payload=build_operational_payload(
            event_id=event_id,
            event_type=event_type,
            booking_id=booking_id,
            mission_anchor_booking_id=mission_anchor_booking_id,
            occurred_at=occurred_at,
            reason=unassign_reason if event_type == EVENT_DRIVER_UNASSIGNED else None,
        ),
    )


def plan_driver_operational_notifications(
    *,
    booking_id: int,
    before_driver_id: int | None,
    after_driver_id: int | None,
    mutation_id: str,
    cancelled: bool = False,
    changes: dict[str, Any] | None = None,
    time_label: str | None = None,
    mission_anchor_booking_id: int | None = None,
    occurred_at: str | None = None,
) -> list[DriverOperationalNotification]:
    """Planifie zéro ou plusieurs pushes pour une seule mutation déjà commitée.

    Annulation : un seul BOOKING_CANCELLED, jamais DRIVER_UNASSIGNED en plus.
    Réattribution : un retrait pour l'ancien, une assignation pour le nouveau.
    Route + horaire dans la même mutation : un seul BOOKING_CHANGED.
    """
    when = occurred_at or datetime.now(UTC).isoformat()
    anchor = mission_anchor_booking_id or booking_id
    before = int(before_driver_id) if before_driver_id else None
    after = int(after_driver_id) if after_driver_id else None

    if cancelled:
        if not before and not after:
            return []
        target = before or after
        if not target:
            return []
        return [
            _notification(
                event_type=EVENT_BOOKING_CANCELLED,
                driver_id=target,
                booking_id=booking_id,
                mutation_id=mutation_id,
                time_label=time_label,
                old_time_label=None,
                new_time_label=None,
                mission_anchor_booking_id=anchor,
                occurred_at=when,
            )
        ]

    planned: list[DriverOperationalNotification] = []
    if before != after:
        if before:
            planned.append(
                _notification(
                    event_type=EVENT_DRIVER_UNASSIGNED,
                    driver_id=before,
                    booking_id=booking_id,
                    mutation_id=mutation_id,
                    time_label=time_label,
                    old_time_label=None,
                    new_time_label=None,
                    mission_anchor_booking_id=anchor,
                    occurred_at=when,
                    unassign_reason=REASON_REASSIGNED if after else REASON_UNASSIGNED,
                )
            )
        if after:
            planned.append(
                _notification(
                    event_type=EVENT_DRIVER_ASSIGNED,
                    driver_id=after,
                    booking_id=booking_id,
                    mutation_id=mutation_id,
                    time_label=time_label,
                    old_time_label=None,
                    new_time_label=None,
                    mission_anchor_booking_id=anchor,
                    occurred_at=when,
                )
            )
        return planned

    if not after:
        return []

    kinds = classify_change_kinds(changes)
    if not kinds:
        return []

    old_time = _change_endpoint(changes, "scheduled_time", "from")
    new_time = _change_endpoint(changes, "scheduled_time", "to") or time_label
    if kinds == {"route", "schedule"}:
        event_type = EVENT_BOOKING_CHANGED
    elif "schedule" in kinds:
        event_type = EVENT_SCHEDULE_CHANGED
    else:
        event_type = EVENT_ROUTE_CHANGED

    return [
        _notification(
            event_type=event_type,
            driver_id=after,
            booking_id=booking_id,
            mutation_id=mutation_id,
            time_label=time_label or new_time,
            old_time_label=old_time,
            new_time_label=new_time,
            mission_anchor_booking_id=anchor,
            occurred_at=when,
        )
    ]
