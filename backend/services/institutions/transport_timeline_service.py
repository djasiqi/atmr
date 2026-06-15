"""Service d'enregistrement et lecture de la timeline transport institution."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ext import db
from models.transport_timeline_event import TransportTimelineEvent

logger = logging.getLogger(__name__)

PAYLOAD_VERSION_DEFAULT = 1

# Champs payload obligatoires par type (autoportance v1)
_SELF_CONTAINED_REQUIRED: dict[str, frozenset[str]] = {
    "offer_sent": frozenset(
        {"company_id", "company_name", "offer_id", "expires_at", "dispatch_mode"}
    ),
    "offer_accepted": frozenset({"company_id", "company_name", "offer_id"}),
    "offer_rejected": frozenset({"company_id", "company_name", "offer_id"}),
    "driver_assigned": frozenset(
        {"driver_id", "driver_name", "company_id", "company_name"}
    ),
    "change_confirmation_requested": frozenset(
        {"change_request_id", "proposed_patch", "before_snapshot"}
    ),
    "change_accepted_by_company": frozenset({"change_request_id"}),
    "change_refused_by_company": frozenset({"change_request_id"}),
    "redispatched": frozenset({"previous_company_id", "previous_company_name"}),
    "external_carrier_assigned": frozenset(
        {"carrier_name", "carrier_phone", "carrier_reference", "reason", "actor_name"}
    ),
    "external_carrier_switched": frozenset(
        {
            "carrier_name",
            "carrier_phone",
            "carrier_reference",
            "reason",
            "actor_name",
            "offers_stopped",
        }
    ),
    "external_mission_completed": frozenset(
        {"carrier_name", "declared_at", "declared_by", "notes"}
    ),
}

_EVENT_LABELS: dict[str, str] = {
    "request_created": "Demande créée",
    "request_sent": "Demande envoyée",
    "offer_sent": "Offre envoyée",
    "offer_rejected": "Offre refusée",
    "offer_expired": "Offre expirée",
    "offer_accepted": "Offre acceptée",
    "request_converted": "Réservation confirmée",
    "booking_created": "Course créée",
    "driver_assigned": "Chauffeur assigné",
    "driver_reassigned": "Chauffeur réassigné",
    "status_changed": "Changement de statut",
    "patient_boarded": "Patient pris en charge",
    "patient_completed": "Transport terminé",
    "field_updated": "Modification",
    "cancelled": "Annulation",
    "billing_changed": "Facturation modifiée",
    "change_confirmation_requested": "Demande de validation envoyée",
    "change_accepted_by_company": "Modification acceptée par le transporteur",
    "change_refused_by_company": "Modification refusée par le transporteur",
    "change_refused_by_driver": "Chauffeur indisponible après modification",
    "driver_reassignment_attempted": "Tentative de réassignation chauffeur",
    "change_expired": "Demande de modification expirée",
    "escalation_required": "Escalade requise — action institution",
    "redispatched": "Course remise en diffusion",
    "route_legs_reorganized": "Parcours modifié",
    "external_carrier_assigned": "Transporteur externe affecté",
    "external_carrier_switched": "Mission basculée vers transporteur externe",
    "external_mission_completed": "Déclarée réalisée par transporteur externe",
}


_FIELD_CHANGE_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "itinéraire",
        frozenset(
            {
                "pickup_location",
                "dropoff_location",
                "dropoff_establishment",
                "dropoff_service",
                "dropoff_doctor",
                "intermediate_stops",
                "multi_stop",
                "return_to_institution",
                "is_round_trip",
                "pickup_lat",
                "pickup_lng",
                "dropoff_lat",
                "dropoff_lng",
            }
        ),
    ),
    (
        "horaires",
        frozenset(
            {
                "mission_date",
                "scheduled_time",
                "scheduled_time_type",
                "pickup_time_confirmed",
                "appointment_time_confirmed",
                "return_time",
                "return_date",
                "return_time_confirmed",
                "return_scheduled_time",
            }
        ),
    ),
    (
        "mobilité",
        frozenset(
            {
                "mobility",
                "requires_wheelchair",
                "requires_assistance",
                "wheelchair_need",
                "wheelchair_client_has",
            }
        ),
    ),
    (
        "notes",
        frozenset({"notes", "notes_medical", "pickup_access_notes", "dropoff_access_notes"}),
    ),
    (
        "patient",
        frozenset({"patient_id", "customer_name", "external_reference"}),
    ),
    (
        "facturation",
        frozenset({"billing_intent", "billing_details"}),
    ),
)


def _summarize_changed_fields(fields: list[str]) -> str:
    """Regroupe les champs techniques en libellés métier lisibles."""
    if not fields:
        return "informations mises à jour"
    field_set = {str(f) for f in fields}
    groups: list[str] = []
    matched: set[str] = set()
    for label, keys in _FIELD_CHANGE_GROUPS:
        if field_set & keys:
            groups.append(label)
            matched |= field_set & keys
    remaining = field_set - matched
    if remaining:
        groups.append("autres détails")
    if len(groups) == 1:
        return groups[0]
    if len(groups) == 2:
        return f"{groups[0]} et {groups[1]}"
    return ", ".join(groups[:-1]) + f" et {groups[-1]}"


def is_timeline_enabled() -> bool:
    return os.getenv("TRANSPORT_TIMELINE_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
    )


def _validate_self_contained(event_type: str, payload: dict[str, Any] | None) -> None:
    required = _SELF_CONTAINED_REQUIRED.get(event_type)
    if not required:
        return
    data = payload or {}
    missing = [k for k in required if data.get(k) is None]
    if missing and os.getenv("FLASK_ENV", "").lower() == "testing":
        raise ValueError(
            f"Timeline payload incomplet pour {event_type}: manque {missing}"
        )
    if missing:
        logger.warning(
            "[TransportTimeline] payload autoportant incomplet event=%s missing=%s",
            event_type,
            missing,
        )


@dataclass(frozen=True, slots=True)
class TimelineActor:
    actor_type: str
    actor_user_id: int | None = None
    company_id: int | None = None
    driver_id: int | None = None


def resolve_actor_name(user_id: int | None) -> str | None:
    """Résout le nom affichable (prénom nom) d'un utilisateur acteur.

    Utilisé pour rendre l'historique traçable (« Parcours modifié — Drin Jasiqi »).
    Retourne ``None`` si l'utilisateur est introuvable ou sans nom.
    """
    if not user_id:
        return None
    try:
        from models.user import User

        user = User.query.get(user_id)
        if not user:
            return None
        name = getattr(user, "full_name", None)
        if name:
            name = name.strip()
        return name or None
    except Exception as resolve_err:  # pragma: no cover - défensif
        logger.warning("[TransportTimeline] resolve_actor_name échec: %s", resolve_err)
        return None


def record_event(
    event_type: str,
    *,
    institution_id: int | None,
    transport_request_id: int | None = None,
    booking_id: int | None = None,
    actor: TimelineActor | None = None,
    payload: dict[str, Any] | None = None,
    payload_version: int = PAYLOAD_VERSION_DEFAULT,
    correlation_id: str | None = None,
    source_event_id: int | None = None,
    commit: bool = False,
) -> TransportTimelineEvent | None:
    """Enregistre un événement timeline (append-only)."""
    if not is_timeline_enabled():
        return None

    _validate_self_contained(event_type, payload)

    if correlation_id:
        existing = TransportTimelineEvent.query.filter_by(
            correlation_id=correlation_id,
            event_type=event_type,
        ).first()
        if existing:
            return existing

    actor = actor or TimelineActor(actor_type="system")
    event = TransportTimelineEvent(
        transport_request_id=transport_request_id,
        booking_id=booking_id,
        institution_id=institution_id,
        event_type=event_type,
        actor_type=actor.actor_type,
        actor_user_id=actor.actor_user_id,
        company_id=actor.company_id,
        driver_id=actor.driver_id,
        payload=payload,
        payload_version=payload_version,
        correlation_id=correlation_id,
        source_event_id=source_event_id,
    )
    db.session.add(event)
    if commit:
        db.session.commit()
    else:
        db.session.flush()
    return event


def build_timeline_label(event: TransportTimelineEvent) -> str:
    base = _EVENT_LABELS.get(event.event_type, event.event_type)
    payload = event.payload or {}

    if event.event_type == "offer_sent":
        name = payload.get("company_name") or ""
        return f"{base} à {name}".strip() if name else base
    if event.event_type in ("offer_accepted", "offer_rejected"):
        name = payload.get("company_name") or ""
        return f"{base} — {name}".strip() if name else base
    if event.event_type == "request_converted":
        name = payload.get("company_name") or ""
        return f"{base} — {name}".strip() if name else base
    if event.event_type == "driver_assigned":
        name = payload.get("driver_name") or ""
        return f"{base} — {name}".strip() if name else base
    if event.event_type == "field_updated":
        fields = payload.get("changed_fields") or []
        if isinstance(fields, dict):
            fields = list(fields.keys())
        if fields:
            summary = _summarize_changed_fields([str(f) for f in fields])
            notified = payload.get("carrier_notified")
            if notified:
                return f"Demande modifiée ({summary}) — transporteur informé"
            return f"Demande modifiée ({summary})"
    if event.event_type == "route_legs_reorganized":
        after = payload.get("after_legs")
        if isinstance(after, list) and after:
            n = len(after)
            suffix = f" — {n} étape{'s' if n > 1 else ''}"
            return f"{base}{suffix}"
    if event.event_type == "status_changed":
        old_s = payload.get("old_status")
        new_s = payload.get("new_status")
        if old_s and new_s:
            return f"{base} : {old_s} → {new_s}"
    return base


def list_timeline_events(
    *,
    institution_id: int,
    transport_request_id: int | None = None,
    booking_id: int | None = None,
    patient_id: int | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    limit: int = 200,
    cursor_id: int | None = None,
) -> list[TransportTimelineEvent]:
    """Liste les événements timeline avec filtres."""
    from models import Booking, TransportRequest

    q = TransportTimelineEvent.query.filter(
        TransportTimelineEvent.institution_id == institution_id
    )

    if transport_request_id is not None:
        q = q.filter(
            TransportTimelineEvent.transport_request_id == transport_request_id
        )
    elif booking_id is not None:
        q = q.filter(TransportTimelineEvent.booking_id == booking_id)
    elif patient_id is not None:
        req_ids = [
            row[0]
            for row in db.session.query(TransportRequest.id)
            .filter_by(institution_id=institution_id, patient_id=patient_id)
            .all()
        ]
        booking_ids = [
            row[0]
            for row in db.session.query(Booking.id)
            .join(TransportRequest, TransportRequest.booking_id == Booking.id)
            .filter(
                TransportRequest.institution_id == institution_id,
                TransportRequest.patient_id == patient_id,
            )
            .all()
        ]
        from sqlalchemy import or_

        filters = []
        if req_ids:
            filters.append(TransportTimelineEvent.transport_request_id.in_(req_ids))
        if booking_ids:
            filters.append(TransportTimelineEvent.booking_id.in_(booking_ids))
        if filters:
            q = q.filter(or_(*filters))
        else:
            return []

    if date_from is not None:
        q = q.filter(TransportTimelineEvent.created_at >= date_from)
    if date_to is not None:
        q = q.filter(TransportTimelineEvent.created_at <= date_to)
    if cursor_id is not None:
        q = q.filter(TransportTimelineEvent.id < cursor_id)

    return (
        q.order_by(TransportTimelineEvent.created_at.desc(), TransportTimelineEvent.id.desc())
        .limit(min(limit, 500))
        .all()
    )


def find_latest_event(
    *,
    transport_request_id: int | None = None,
    booking_id: int | None = None,
    event_type: str,
    company_id: int | None = None,
) -> TransportTimelineEvent | None:
    """Trouve le dernier event d'un type (pour source_event_id)."""
    q = TransportTimelineEvent.query.filter_by(event_type=event_type)
    if transport_request_id is not None:
        q = q.filter_by(transport_request_id=transport_request_id)
    if booking_id is not None:
        q = q.filter_by(booking_id=booking_id)
    if company_id is not None:
        q = q.filter_by(company_id=company_id)
    return q.order_by(TransportTimelineEvent.id.desc()).first()
