"""Service métier : audit trail, versioning et alertes critiques institution."""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from application.bookings.cancellation_rules import (
    compute_cancellation_fee,
    get_cancellation_display_label,
)
from ext import db
from models import (
    Booking,
    BookingChangeAcknowledgement,
    BookingChangeEvent,
    BookingChangeRequest,
    TransportRequest,
)
from models.booking_change_request import (
    TransportActionStatus,
)
from models.enums import BookingStatus, InstitutionRole
from shared.time_utils import normalize_mission_wall_clock

logger = logging.getLogger(__name__)

MIN_CRITICAL_REASON_LEN = 10

# Durée de vie par défaut d'une demande de validation transporteur (en minutes).
DEFAULT_CHANGE_REQUEST_TTL_MINUTES = 120

# Statuts booking pour lesquels un transporteur est « engagé » mais la course
# n'a pas encore démarré : une modification critique doit être révalidée.
COMMITTED_STATUSES = frozenset(
    {
        BookingStatus.ACCEPTED.value,
        BookingStatus.ASSIGNED.value,
    }
)


def is_revalidation_enabled() -> bool:
    """Feature flag : révalidation transporteur après modification critique.

    Activé par défaut (dev) ; désactivable via
    INSTITUTION_CHANGE_REVALIDATION_ENABLED=false.
    """
    return os.getenv("INSTITUTION_CHANGE_REVALIDATION_ENABLED", "true").lower() in (
        "1",
        "true",
        "yes",
    )


def get_change_request_ttl_minutes() -> int:
    """TTL (minutes) avant expiration d'une demande de validation transporteur."""
    raw = os.getenv("INSTITUTION_CHANGE_REQUEST_TTL_MINUTES")
    if not raw:
        return DEFAULT_CHANGE_REQUEST_TTL_MINUTES
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_CHANGE_REQUEST_TTL_MINUTES
    except (TypeError, ValueError):
        return DEFAULT_CHANGE_REQUEST_TTL_MINUTES


# Champs « critiques » dont la modification après acceptation transporteur
# nécessite une révalidation (réutilise MAJOR_FIELDS, défini plus bas).
def _revalidation_trigger_fields() -> frozenset[str]:
    return MAJOR_FIELDS


LEG_SCHEDULE_PATCH_FIELDS = frozenset(
    {
        "appointment_time",
        "leg_appointments",
        "return_appointment_time",
    }
)

INSTITUTION_OPERATIONAL_FIELDS = frozenset(
    {
        "customer_name",
        "pickup_location",
        "dropoff_location",
        "pickup_lat",
        "pickup_lon",
        "dropoff_lat",
        "dropoff_lon",
        "scheduled_time",
        "medical_facility",
        "doctor_name",
        "hospital_service",
        "notes_medical",
        "pickup_access_notes",
        "dropoff_access_notes",
        "pickup_floor",
        "pickup_door_code",
        "dropoff_floor",
        "dropoff_door_code",
        "wheelchair_client_has",
        "wheelchair_need",
        "mission_type",
        "delivery_description",
    }
)

CRITICAL_EN_ROUTE_FIELDS = frozenset(
    {
        "pickup_location",
        "dropoff_location",
        "pickup_lat",
        "pickup_lon",
        "dropoff_lat",
        "dropoff_lon",
        "wheelchair_client_has",
        "wheelchair_need",
    }
)

MAJOR_FIELDS = frozenset(
    {
        "scheduled_time",
        "pickup_location",
        "dropoff_location",
        "pickup_lat",
        "pickup_lon",
        "dropoff_lat",
        "dropoff_lon",
        "wheelchair_client_has",
        "wheelchair_need",
        "mission_type",
        "delivery_description",
        "customer_name",
    }
)

BILLING_CHANGE_REASON_CODES = frozenset(
    {
        "PRICE_CORRECTION",
        "INSTITUTION_TAKEOVER",
        "PATIENT_REQUEST",
        "INSURANCE_TAKEOVER",
        "ADMIN_CORRECTION",
        "OTHER",
    }
)

OPERATIONAL_ROLES = frozenset(
    {
        InstitutionRole.ADMIN.value,
        InstitutionRole.REQUESTER.value,
        InstitutionRole.BILLING.value,
        InstitutionRole.CURATOR.value,
    }
)


def _status_value(status: Any) -> str:
    if status is None:
        return ""
    return str(getattr(status, "value", status)).upper()


def _booking_operational_snapshot(booking: Booking) -> dict[str, Any]:
    st = booking.scheduled_time
    return {
        "customer_name": booking.customer_name,
        "pickup_location": booking.pickup_location,
        "dropoff_location": booking.dropoff_location,
        "pickup_lat": booking.pickup_lat,
        "pickup_lon": booking.pickup_lon,
        "dropoff_lat": booking.dropoff_lat,
        "dropoff_lon": booking.dropoff_lon,
        "scheduled_time": st.isoformat() if st else None,
        "medical_facility": booking.medical_facility,
        "doctor_name": booking.doctor_name,
        "hospital_service": booking.hospital_service,
        "notes_medical": booking.notes_medical,
        "pickup_access_notes": booking.pickup_access_notes,
        "dropoff_access_notes": booking.dropoff_access_notes,
        "pickup_floor": booking.pickup_floor,
        "pickup_door_code": booking.pickup_door_code,
        "dropoff_floor": booking.dropoff_floor,
        "dropoff_door_code": booking.dropoff_door_code,
        "wheelchair_client_has": bool(booking.wheelchair_client_has),
        "wheelchair_need": bool(booking.wheelchair_need),
        "mission_type": booking.mission_type,
        "delivery_description": booking.delivery_description,
        "status": _status_value(booking.status),
        "boarded_at": booking.boarded_at.isoformat() if booking.boarded_at else None,
        "edit_version": int(booking.edit_version or 1),
    }


def _billing_snapshot(booking: Booking) -> dict[str, Any]:
    return {
        "billed_to_type": booking.billed_to_type,
        "billed_to_company_id": booking.billed_to_company_id,
        "billed_to_contact": booking.billed_to_contact,
        "billing_override_reason": booking.billing_override_reason,
        "amount": float(booking.amount) if booking.amount is not None else None,
        "edit_version": int(booking.edit_version or 1),
    }


def _changed_fields_map(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, bool]:
    keys = set(before) | set(after)
    return {
        k: before.get(k) != after.get(k) for k in keys if before.get(k) != after.get(k)
    }


def classify_change(
    changed: set[str],
    *,
    is_en_route: bool,
    is_cancellation: bool = False,
) -> tuple[str, str, bool]:
    """Retourne (change_class, severity, ack_required)."""
    if is_cancellation and is_en_route:
        return "critical", "CRITICAL", True
    if is_cancellation:
        return "major", "WARNING", False

    critical_hits = changed & CRITICAL_EN_ROUTE_FIELDS
    if is_en_route and critical_hits:
        return "critical", "CRITICAL", True

    if changed & MAJOR_FIELDS:
        sev = "WARNING" if is_en_route else "INFO"
        return "major", sev, False

    if changed:
        return "minor", "INFO", False
    return "minor", "INFO", False


def build_operational_impact(
    *,
    change_class: str,
    severity: str,
    ack_required: bool,
    is_en_route: bool,
    changed_fields: set[str],
) -> dict[str, Any]:
    return {
        "fanout_company": severity in ("WARNING", "CRITICAL"),
        "fanout_driver": severity == "CRITICAL",
        "ack_required": ack_required,
        "recalculate_route": bool(
            is_en_route
            and changed_fields
            & {
                "pickup_location",
                "dropoff_location",
                "pickup_lat",
                "pickup_lon",
                "dropoff_lat",
                "dropoff_lon",
            }
        ),
        "billing_review": change_class == "critical",
        "en_route": is_en_route,
    }


@dataclass(frozen=True, slots=True)
class InstitutionBookingContext:
    booking: Booking
    transport_request: TransportRequest
    institution_id: int


def resolve_institution_booking(
    booking_id: int, institution_id: int
) -> InstitutionBookingContext | None:
    transport_req = TransportRequest.query.filter_by(
        booking_id=booking_id,
        institution_id=institution_id,
    ).first()
    if not transport_req:
        return None
    booking = Booking.query.get(booking_id)
    if not booking:
        return None
    return InstitutionBookingContext(
        booking=booking,
        transport_request=transport_req,
        institution_id=institution_id,
    )


def assert_operational_role(role: str | None) -> str | None:
    if role not in OPERATIONAL_ROLES:
        return (
            f"Rôle requis: {', '.join(sorted(OPERATIONAL_ROLES))}. Votre rôle: {role}"
        )
    return None


def assert_not_boarded(booking: Booking) -> str | None:
    if booking.boarded_at is not None:
        return (
            "Modification impossible : le patient est déjà pris en charge "
            f"(boarded_at={booking.boarded_at.isoformat()})."
        )
    return None


def check_version(
    booking: Booking, client_version: int | None
) -> dict[str, Any] | None:
    current = int(booking.edit_version or 1)
    if client_version is None:
        return {"error": "version requise pour la mise à jour optimiste."}
    if int(client_version) != current:
        return {
            "error": "Conflit de version : la réservation a été modifiée entre-temps.",
            "current_version": current,
            "current_snapshot": _booking_operational_snapshot(booking),
        }
    return None


def record_change_event(
    *,
    booking: Booking,
    transport_request: TransportRequest | None,
    institution_id: int | None,
    actor_user_id: int | None,
    actor_role: str | None,
    actor_type: str,
    actor_display_name: str | None,
    action_type: str,
    change_scope: str,
    source: str,
    before_snapshot: dict[str, Any] | None,
    after_snapshot: dict[str, Any] | None,
    reason: str | None = None,
    change_class: str = "minor",
    severity: str = "INFO",
    ack_required: bool = False,
    operational_impact: dict[str, Any] | None = None,
    financial_actor_role: str | None = None,
    billing_change_reason_code: str | None = None,
    correlation_id: str | None = None,
) -> BookingChangeEvent:
    changed_fields = None
    if before_snapshot and after_snapshot:
        changed_fields = _changed_fields_map(before_snapshot, after_snapshot)

    event = BookingChangeEvent()
    event.booking_id = booking.id
    event.transport_request_id = transport_request.id if transport_request else None
    event.institution_id = institution_id
    event.booking_version = int(booking.edit_version or 1)
    event.actor_user_id = actor_user_id
    event.actor_role = actor_role
    event.actor_type = actor_type
    event.actor_display_name = actor_display_name
    event.action_type = action_type
    event.change_class = change_class
    event.severity = severity
    event.before_snapshot = before_snapshot
    event.after_snapshot = after_snapshot
    event.changed_fields = changed_fields
    event.reason = reason
    event.change_scope = change_scope
    event.source = source
    event.operational_impact = operational_impact
    event.financial_actor_role = financial_actor_role
    event.billing_change_reason_code = billing_change_reason_code
    event.ack_required = ack_required
    event.correlation_id = correlation_id or str(uuid.uuid4())
    db.session.add(event)

    # Dual-write timeline transport pour les modifications/annulations opérationnelles
    if action_type in ("field_updated", "cancelled"):
        _record_change_timeline(
            booking=booking,
            transport_request=transport_request,
            institution_id=institution_id,
            actor_user_id=actor_user_id,
            actor_type=actor_type,
            actor_display_name=actor_display_name,
            action_type=action_type,
            changed_fields=changed_fields,
            reason=reason,
            correlation_id=event.correlation_id,
        )

    return event


def _record_change_timeline(
    *,
    booking: Booking,
    transport_request: TransportRequest | None,
    institution_id: int | None,
    actor_user_id: int | None,
    actor_type: str,
    actor_display_name: str | None,
    action_type: str,
    changed_fields: dict[str, bool] | None,
    reason: str | None,
    correlation_id: str | None,
) -> None:
    """Réplique une modification/annulation booking dans la timeline transport."""
    try:
        from services.institutions.transport_timeline_service import (
            TimelineActor,
            record_event,
            resolve_actor_name,
        )

        actor_name = actor_display_name or resolve_actor_name(actor_user_id)
        payload: dict[str, Any] = {
            "changed_fields": changed_fields,
            "reason": reason,
            "actor_name": actor_name,
        }
        if action_type == "cancelled":
            payload["cancellation_display_label"] = getattr(
                booking, "cancellation_display_label", None
            )
            payload["cancelled_by_role"] = actor_type

        record_event(
            action_type,
            institution_id=institution_id,
            transport_request_id=transport_request.id if transport_request else None,
            booking_id=booking.id,
            actor=TimelineActor(actor_type=actor_type, actor_user_id=actor_user_id),
            payload=payload,
            correlation_id=f"{action_type}:{correlation_id}",
        )
    except Exception as timeline_err:
        logger.warning("[BookingChange] Timeline recording failed: %s", timeline_err)


def bump_edit_version(booking: Booking) -> int:
    booking.edit_version = int(booking.edit_version or 1) + 1
    booking.updated_at = datetime.now(UTC)
    return booking.edit_version


def apply_operational_patch(booking: Booking, validated: dict[str, Any]) -> list[str]:
    updated: list[str] = []
    for key, value in validated.items():
        if key not in INSTITUTION_OPERATIONAL_FIELDS:
            continue
        if key == "scheduled_time":
            # Règle d'architecture : écriture mission institution → normalize_mission_wall_clock.
            # parse_local_naive reste autorisé côté booking entreprise (ManualBookingForm).
            scheduled_local = normalize_mission_wall_clock(value)
            if scheduled_local is None:
                raise ValueError("Heure planifiée invalide.")
            booking.scheduled_time = scheduled_local
            if hasattr(booking, "time_confirmed"):
                if "time_confirmed" in validated:
                    booking.time_confirmed = bool(validated["time_confirmed"])
                else:
                    is_midnight = (
                        scheduled_local.hour == 0
                        and scheduled_local.minute == 0
                        and scheduled_local.second == 0
                    )
                    booking.time_confirmed = not is_midnight
                updated.append("time_confirmed")
        else:
            setattr(booking, key, value)
        updated.append(key)

    if "scheduled_time" in updated:
        from services.institutions.mission_schedule import (
            sync_request_departure_for_booking,
        )

        if sync_request_departure_for_booking(booking):
            updated.append("transport_request.scheduled_time")

    return updated


def sync_transport_request_leg_schedule(
    transport_request: TransportRequest | None,
    booking: Booking,
    *,
    appointment_time: str | None = None,
    leg_appointments: list[dict[str, Any]] | None = None,
    return_appointment_time: str | None = None,
) -> list[str]:
    """Met à jour les heures RDV sur les legs liés à la demande convertie."""
    if transport_request is None:
        return []

    from models.transport_request_leg import TransportRequestLeg

    legs = (
        TransportRequestLeg.query.filter_by(transport_request_id=transport_request.id)
        .order_by(TransportRequestLeg.sequence_index.asc())
        .all()
    )
    if not legs:
        return []

    updated: list[str] = []
    has_return = bool(getattr(transport_request, "return_to_institution", False))
    dest_legs = legs[:-1] if has_return and len(legs) > 1 else legs
    return_leg = legs[-1] if has_return and len(legs) > 1 else None

    def _apply_leg_time(leg: TransportRequestLeg, iso: str | None, label: str) -> None:
        if iso:
            # Règle d'architecture : écriture mission institution → normalize_mission_wall_clock.
            parsed = normalize_mission_wall_clock(iso)
            if parsed is None:
                raise ValueError("Heure de rendez-vous invalide.")
            leg.scheduled_time = parsed
            leg.time_confirmed = True
        else:
            leg.scheduled_time = None
            leg.time_confirmed = False
        updated.append(label)

    if leg_appointments:
        for item in leg_appointments:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            iso = item.get("scheduled_time")
            if idx is None or not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(dest_legs):
                continue
            _apply_leg_time(dest_legs[idx], iso, f"leg[{idx}].scheduled_time")
    elif appointment_time is not None:
        leg = next((leg for leg in dest_legs if leg.booking_id == booking.id), None)
        if leg is None and dest_legs:
            leg = dest_legs[0]
        if leg is not None:
            _apply_leg_time(leg, appointment_time, "leg[0].scheduled_time")

    if return_appointment_time is not None and return_leg is not None:
        _apply_leg_time(
            return_leg, return_appointment_time, "leg.return.scheduled_time"
        )
        if return_appointment_time:
            # Règle d'architecture : écriture mission institution → normalize_mission_wall_clock.
            parsed_return = normalize_mission_wall_clock(return_appointment_time)
            if parsed_return is not None:
                transport_request.return_time = parsed_return
                transport_request.return_time_confirmed = True
                updated.append("return_time")
        else:
            transport_request.return_time = None
            transport_request.return_time_confirmed = False
            updated.append("return_time")

    return updated


def _company_is_committed(booking: Booking, status: str) -> bool:
    """Un transporteur est engagé (course pas encore démarrée)."""
    has_company = bool(booking.company_id or booking.executing_company_id)
    return has_company and status in COMMITTED_STATUSES


def _simulate_after_snapshot(booking: Booking, patch: dict[str, Any]) -> dict[str, Any]:
    """Calcule un snapshot opérationnel « après patch » sans muter le booking."""
    after = _booking_operational_snapshot(booking)
    for key, value in patch.items():
        if key not in INSTITUTION_OPERATIONAL_FIELDS:
            continue
        if key == "scheduled_time":
            parsed = normalize_mission_wall_clock(value)
            after["scheduled_time"] = parsed.isoformat() if parsed else value
        elif key in ("wheelchair_client_has", "wheelchair_need"):
            after[key] = bool(value)
        else:
            after[key] = value
    return after


def supersede_pending_change_requests(
    booking: Booking,
    *,
    excluding_id: int | None = None,
) -> int:
    """Ferme les TransportActions ouvertes (CLOSED_REPLACED)."""
    from application.institutions.transport_action_workflow import (
        close_open_actions_as_replaced,
    )

    return len(close_open_actions_as_replaced(booking, excluding_id=excluding_id))


def _notify_company_change_request(
    booking: Booking,
    change_request: BookingChangeRequest,
) -> None:
    """Notifie l'entreprise qu'une validation de modification est requise."""
    try:
        from services.events.institution_events import persist_company_notification

        company_id = booking.company_id or booking.executing_company_id
        if not company_id:
            return
        patient = booking.customer_name or "Patient"
        fields = list((change_request.changed_fields or {}).keys())
        is_cancel = (change_request.action_type or "") == "CANCELLATION"
        title = "Annulation à confirmer" if is_cancel else "Modification à confirmer"
        msg = (
            f"{'Demande d’annulation' if is_cancel else 'Modification demandée'} — "
            f"course #{booking.id} ({patient}). "
            f"La course reste active jusqu’à votre décision."
            + (f" Champs : {', '.join(fields)}." if fields and not is_cancel else "")
        )
        persist_company_notification(
            company_id=int(company_id),
            event_type="institution_change_request",
            title=title,
            message=msg,
            metadata={
                "booking_id": booking.id,
                "change_request_id": change_request.id,
                "action_type": change_request.action_type
                or ("CANCELLATION" if is_cancel else "CHANGE"),
                "changed_fields": change_request.changed_fields,
                "expires_at": change_request.expires_at.isoformat()
                if change_request.expires_at
                else None,
            },
            dedupe_key=f"inst_change_req_{change_request.id}",
        )
    except Exception as notif_err:
        logger.warning(
            "[BookingChange] notify company change_request failed: %s", notif_err
        )


def create_change_request(
    ctx: InstitutionBookingContext,
    *,
    patch: dict[str, Any],
    reason: str | None,
    actor_user_id: int | None,
    actor_role: str | None,
    actor_display_name: str | None,
) -> tuple[dict[str, Any], int]:
    """Crée une demande de validation transporteur au lieu d'appliquer le patch.

    - Supersede les BCR PENDING existantes
    - Met à jour booking.active_change_request_id
    - Historise un événement timeline change_confirmation_requested
    """
    booking = ctx.booking
    before = _booking_operational_snapshot(booking)
    after = _simulate_after_snapshot(booking, patch)
    changed_fields = _changed_fields_map(before, after)

    if not changed_fields:
        return {"error": "Aucun champ modifié."}, 400

    from application.institutions.transport_action_workflow import (
        classify_action_type,
        create_transport_action_from_intention,
    )

    change_request = create_transport_action_from_intention(
        booking=booking,
        transport_request=ctx.transport_request,
        institution_id=ctx.institution_id,
        action_type=classify_action_type(changed_fields),
        proposed_patch=patch,
        before_snapshot=before,
        after_snapshot=after,
        changed_fields=changed_fields,
        reason=reason or None,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
    )

    record_change_event(
        booking=booking,
        transport_request=ctx.transport_request,
        institution_id=ctx.institution_id,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_type="institution_user",
        actor_display_name=actor_display_name,
        action_type="change_request_created",
        change_scope="operational",
        source="institution_portal",
        before_snapshot=before,
        after_snapshot=after,
        reason=reason or None,
        change_class="major",
        severity="WARNING",
        ack_required=False,
        operational_impact={"revalidation_required": True},
    )

    _record_change_request_timeline(ctx=ctx, change_request=change_request)
    db.session.flush()

    _notify_company_change_request(booking, change_request)

    db.session.commit()
    try:
        from domain.events.events import TransportActionRequestedEvent
        from shared.events.event_bus import publish_event

        publish_event(
            TransportActionRequestedEvent(
                action_id=change_request.id,
                booking_id=booking.id,
                action_type=change_request.action_type or "CHANGE_OTHER",
                institution_id=ctx.institution_id,
            )
        )
    except Exception as exc:
        logger.warning("[BookingChange] publish Requested: %s", exc)

    return {
        "success": True,
        "status": "pending_action",
        "pending_revalidation": True,
        "booking_id": booking.id,
        "edit_version": int(booking.edit_version or 1),
        "change_request": change_request.serialize(),
    }, 202


def _record_change_request_timeline(
    *,
    ctx: InstitutionBookingContext,
    change_request: BookingChangeRequest,
) -> None:
    """Historise change_confirmation_requested dans la timeline transport."""
    try:
        from services.institutions.transport_timeline_service import (
            TimelineActor,
            find_latest_event,
            record_event,
            resolve_actor_name,
        )

        source = find_latest_event(
            booking_id=ctx.booking.id,
            event_type="booking_created",
        )
        record_event(
            "change_confirmation_requested",
            institution_id=ctx.institution_id,
            transport_request_id=(
                ctx.transport_request.id if ctx.transport_request else None
            ),
            booking_id=ctx.booking.id,
            actor=TimelineActor(
                actor_type="institution_user",
                actor_user_id=change_request.requested_by_user_id,
            ),
            payload={
                "change_request_id": change_request.id,
                "action_type": change_request.action_type,
                "proposed_patch": change_request.proposed_patch,
                "before_snapshot": change_request.before_snapshot,
                "after_snapshot": change_request.after_snapshot,
                "changed_fields": change_request.changed_fields,
                "reason": change_request.reason,
                "actor_name": resolve_actor_name(change_request.requested_by_user_id),
            },
            correlation_id=f"change_confirmation_requested:{change_request.id}",
            source_event_id=source.id if source else None,
        )
    except Exception as timeline_err:
        logger.warning(
            "[BookingChange] change_request timeline failed: %s", timeline_err
        )


def fanout_critical_change(
    booking: Booking,
    event: BookingChangeEvent,
    transport_request: TransportRequest | None,
) -> None:
    """Notifications haute priorité entreprise + chauffeur."""
    from services.events.fanout import send_critical_alert_ios
    from services.events.institution_events import persist_company_notification

    company_id = booking.company_id or booking.executing_company_id
    req_id = transport_request.id if transport_request else None
    public_id = (
        getattr(transport_request, "public_id", None) if transport_request else None
    )
    patient = booking.customer_name or "Patient"
    msg = f"Modification institution en route — {patient} (course #{booking.id})"
    meta = {
        "booking_id": booking.id,
        "request_id": req_id,
        "public_id": public_id,
        "event_id": event.id,
        "severity": event.severity,
        "changed_fields": event.changed_fields,
    }

    if company_id:
        persist_company_notification(
            company_id=int(company_id),
            event_type="institution_booking_critical_change",
            title="Modification institution (en route)",
            message=msg,
            metadata=meta,
            dedupe_key=f"inst_crit_{booking.id}_{event.correlation_id}",
        )

    driver_id = booking.driver_id
    if driver_id:
        send_critical_alert_ios(
            driver_id=int(driver_id),
            title="Course modifiée",
            message=msg,
            alert_type="institution_booking_change",
            data=meta,
        )

    record_change_event(
        booking=booking,
        transport_request=transport_request,
        institution_id=event.institution_id,
        actor_user_id=event.actor_user_id,
        actor_type="system",
        actor_role=None,
        actor_display_name="Système",
        action_type="notification_sent",
        change_scope="operational",
        source="system",
        before_snapshot=None,
        after_snapshot={"event_id": event.id, "channels": ["company", "driver"]},
        change_class="minor",
        severity="INFO",
        ack_required=False,
        correlation_id=event.correlation_id,
    )


def _collect_linked_bookings(booking: Booking) -> list[Booking]:
    """Retourne les bookings liés à annuler en cascade avec `booking`.

    Couvre deux cas :
      - parcours multi-destinations : tous les legs partageant le même
        `route_group_id` (ex. A→B, B→C, C→A) ;
      - aller-retour classique : la course retour rattachée via
        `parent_booking_id`.

    Le booking source n'est jamais inclus dans la liste retournée.
    """
    linked: dict[int, Booking] = {}

    route_group_id = getattr(booking, "route_group_id", None)
    if route_group_id:
        siblings = Booking.query.filter(
            Booking.route_group_id == route_group_id,
            Booking.id != booking.id,
        ).all()
        for sib in siblings:
            linked[sib.id] = sib

    # Retours rattachés (A/R) au booking source ou à un de ses legs.
    parent_ids = {booking.id, *linked.keys()}
    returns = Booking.query.filter(
        Booking.parent_booking_id.in_(parent_ids),
        Booking.id != booking.id,
    ).all()
    for ret in returns:
        linked[ret.id] = ret

    return list(linked.values())


def cancel_institution_booking(
    ctx: InstitutionBookingContext,
    *,
    reason: str,
    reason_code: str | None,
    actor_user_id: int | None,
    actor_role: str | None,
    actor_display_name: str | None,
    client_version: int,
) -> tuple[dict[str, Any], int]:
    """Post-engagement : crée une TransportAction CANCELLATION (pas d'annulation immédiate).

    Pré-engagement : annulation directe conservée.
    IN_PROGRESS : 422 INTERRUPTION_REQUIRED.
    """
    booking = ctx.booking
    err = assert_not_boarded(booking)
    if err:
        return {"error": err}, 400

    conflict = check_version(booking, client_version)
    if conflict:
        return conflict, 409

    status = _status_value(booking.status)
    if status in ("COMPLETED", "RETURN_COMPLETED"):
        return {"error": "Course terminée, annulation impossible."}, 400

    if status == BookingStatus.IN_PROGRESS.value:
        return {
            "error": "Transport en cours : demandez une interruption, pas une annulation.",
            "code": "INTERRUPTION_REQUIRED",
        }, 422

    is_en_route = status == BookingStatus.EN_ROUTE.value
    if is_en_route and len((reason or "").strip()) < MIN_CRITICAL_REASON_LEN:
        return {
            "error": f"Motif obligatoire (min. {MIN_CRITICAL_REASON_LEN} caractères) pour annulation en route.",
        }, 400

    # Transporteur engagé → intention d'annulation (V1.1)
    if _company_is_committed(booking, status) or is_en_route:
        from application.institutions.transport_action_workflow import (
            classify_action_type,
            create_transport_action_from_intention,
        )

        before = _booking_operational_snapshot(booking)
        after = {**before, "status": "CANCELED"}
        action = create_transport_action_from_intention(
            booking=booking,
            transport_request=ctx.transport_request,
            institution_id=ctx.institution_id,
            action_type=classify_action_type(set(), is_cancellation=True),
            proposed_patch={
                "_cancellation": True,
                "reason_code": reason_code or "CLIENT_REQUEST",
            },
            before_snapshot=before,
            after_snapshot=after,
            changed_fields={"status": True},
            reason=reason,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            action_scope="ROUND_TRIP"
            if getattr(booking, "parent_booking_id", None)
            or getattr(booking, "route_group_id", None)
            else "BOOKING",
        )
        record_change_event(
            booking=booking,
            transport_request=ctx.transport_request,
            institution_id=ctx.institution_id,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            actor_type="institution_user",
            actor_display_name=actor_display_name,
            action_type="change_request_created",
            change_scope="cancellation",
            source="institution_portal",
            before_snapshot=before,
            after_snapshot=after,
            reason=reason,
            change_class="major",
            severity="CRITICAL" if is_en_route else "WARNING",
            ack_required=False,
            operational_impact={
                "cancellation_requested": True,
                "mission_unchanged": True,
            },
        )
        _record_change_request_timeline(ctx=ctx, change_request=action)
        _notify_company_change_request(booking, action)
        db.session.commit()
        try:
            from domain.events.events import TransportActionRequestedEvent
            from shared.events.event_bus import publish_event

            publish_event(
                TransportActionRequestedEvent(
                    action_id=action.id,
                    booking_id=booking.id,
                    action_type=action.action_type or "CANCELLATION",
                    institution_id=ctx.institution_id,
                )
            )
        except Exception as exc:
            logger.warning("[InstitutionCancel] publish Requested: %s", exc)
        return {
            "success": True,
            "status": "pending_action",
            "pending_revalidation": True,
            "booking_id": booking.id,
            "edit_version": int(booking.edit_version or 1),
            "change_request": action.serialize(),
            "message": "Demande d'annulation envoyée au transporteur. La course reste active jusqu'à confirmation.",
        }, 202

    # Pré-engagement : annulation directe (legacy path)
    before = _booking_operational_snapshot(booking)
    cancelled_at = datetime.now(UTC)

    from models.invoice import CompanyBillingSettings

    billing = (
        CompanyBillingSettings.query.filter_by(company_id=booking.company_id).first()
        if getattr(booking, "company_id", None)
        else None
    )
    cancellation_policy = (
        getattr(billing, "cancellation_policy", None) if billing else None
    )

    fee = compute_cancellation_fee(
        booking,
        status_at_cancel=status,
        cancelled_at=cancelled_at,
        reason_code=reason_code or "CLIENT_REQUEST",
        policy=cancellation_policy,
    )
    is_status_forced_billable = status in (
        BookingStatus.EN_ROUTE.value,
        BookingStatus.IN_PROGRESS.value,
    )
    is_billable = True if is_status_forced_billable else fee.is_billable

    booking.status = BookingStatus.CANCELED
    booking.cancellation_reason_code = reason_code or "CLIENT_REQUEST"
    booking.cancellation_reason_text = reason
    booking.is_cancellation_billable = is_billable
    booking.cancellation_display_label = get_cancellation_display_label(
        booking.cancellation_reason_code, reason
    )
    if fee.fee_amount is not None:
        booking.cancellation_fee_amount = fee.fee_amount
    booking.cancellation_fee_percent = fee.percent
    booking.cancellation_fee_tier_id = fee.tier_id

    change_class, severity, ack_required = classify_change(
        set(), is_en_route=is_en_route, is_cancellation=True
    )
    impact = build_operational_impact(
        change_class=change_class,
        severity=severity,
        ack_required=ack_required,
        is_en_route=is_en_route,
        changed_fields=set(),
    )
    bump_edit_version(booking)
    after = _booking_operational_snapshot(booking)

    event = record_change_event(
        booking=booking,
        transport_request=ctx.transport_request,
        institution_id=ctx.institution_id,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_type="institution_user",
        actor_display_name=actor_display_name,
        action_type="cancelled",
        change_scope="cancellation",
        source="institution_portal",
        before_snapshot=before,
        after_snapshot=after,
        reason=reason,
        change_class=change_class,
        severity=severity,
        ack_required=ack_required,
        operational_impact=impact,
    )
    db.session.flush()

    if ack_required:
        fanout_critical_change(booking, event, ctx.transport_request)

    # ── Cascade : annuler les legs liés (multi-destinations) et retours A/R ──
    # La facturation est portée par le booking principal ; les legs/retours liés
    # sont marqués NON facturables pour éviter une double facturation (cohérent
    # avec la cascade aller → retour).
    cancelled_linked_ids: list[int] = []
    terminal_statuses = {"COMPLETED", "RETURN_COMPLETED", "CANCELED"}
    for linked in _collect_linked_bookings(booking):
        if _status_value(linked.status) in terminal_statuses:
            continue
        linked_before = _booking_operational_snapshot(linked)
        linked.status = BookingStatus.CANCELED
        linked.cancellation_reason_code = booking.cancellation_reason_code
        linked.cancellation_reason_text = reason
        linked.is_cancellation_billable = False
        linked.cancellation_display_label = booking.cancellation_display_label
        bump_edit_version(linked)
        linked_after = _booking_operational_snapshot(linked)
        record_change_event(
            booking=linked,
            transport_request=ctx.transport_request,
            institution_id=ctx.institution_id,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            actor_type="institution_user",
            actor_display_name=actor_display_name,
            action_type="cancelled",
            change_scope="cancellation",
            source="institution_portal",
            before_snapshot=linked_before,
            after_snapshot=linked_after,
            reason=reason,
            change_class=change_class,
            severity=severity,
            ack_required=ack_required,
            operational_impact=impact,
        )
        cancelled_linked_ids.append(linked.id)

    if cancelled_linked_ids:
        db.session.flush()
        logger.info(
            "[InstitutionCancel] Cascade annulation booking=%s, legs/retours liés=%s",
            booking.id,
            cancelled_linked_ids,
        )

    db.session.commit()
    return {
        "success": True,
        "booking_id": booking.id,
        "status": _status_value(booking.status),
        "is_cancellation_billable": is_billable,
        "edit_version": booking.edit_version,
        "change_event_id": event.id,
        "cancelled_linked_booking_ids": cancelled_linked_ids,
    }, 200


def update_institution_booking(
    ctx: InstitutionBookingContext,
    *,
    payload: dict[str, Any],
    actor_user_id: int | None,
    actor_role: str | None,
    actor_display_name: str | None,
) -> tuple[dict[str, Any], int]:
    booking = ctx.booking
    unknown = (
        set(payload.keys())
        - INSTITUTION_OPERATIONAL_FIELDS
        - {"version", "reason"}
        - LEG_SCHEDULE_PATCH_FIELDS
    )
    if unknown:
        return {
            "error": "Champs non autorisés.",
            "rejected_fields": sorted(unknown),
        }, 400

    patch = {k: v for k, v in payload.items() if k in INSTITUTION_OPERATIONAL_FIELDS}
    leg_schedule_present = any(
        payload.get(k) is not None for k in LEG_SCHEDULE_PATCH_FIELDS
    )
    if not patch and not leg_schedule_present:
        return {"error": "Aucun champ opérationnel à mettre à jour."}, 400

    err = assert_not_boarded(booking)
    if err:
        return {"error": err}, 400

    client_version = payload.get("version")
    conflict = check_version(booking, client_version)
    if conflict:
        return conflict, 409

    status = _status_value(booking.status)
    if status in ("COMPLETED", "RETURN_COMPLETED", "CANCELED"):
        return {"error": f"Modification impossible (statut {status})."}, 400

    is_en_route = status == BookingStatus.EN_ROUTE.value
    reason = (payload.get("reason") or "").strip()
    changed_preview = set(patch.keys())
    _cc, _sev, ack_required = classify_change(changed_preview, is_en_route=is_en_route)
    if (
        is_en_route
        and (changed_preview & CRITICAL_EN_ROUTE_FIELDS)
        and len(reason) < MIN_CRITICAL_REASON_LEN
    ):
        return {
            "error": f"Motif obligatoire (min. {MIN_CRITICAL_REASON_LEN} caractères) pour modification en route.",
        }, 400

    # Modèle strict V1.1 : toute modification opérationnelle post-engagement
    # → TransportAction (pas de patch direct).
    needs_revalidation = is_revalidation_enabled() and _company_is_committed(
        booking, status
    )
    if needs_revalidation:
        # Inclure les champs de planning legs dans le patch soumis à décision
        full_patch = dict(patch)
        for k in LEG_SCHEDULE_PATCH_FIELDS:
            if payload.get(k) is not None:
                full_patch[k] = payload.get(k)
        return create_change_request(
            ctx,
            patch=full_patch or patch,
            reason=reason or None,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            actor_display_name=actor_display_name,
        )

    before = _booking_operational_snapshot(booking)
    try:
        updated_fields = apply_operational_patch(booking, patch) if patch else []
        leg_updated = sync_transport_request_leg_schedule(
            ctx.transport_request,
            booking,
            appointment_time=payload.get("appointment_time"),
            leg_appointments=payload.get("leg_appointments"),
            return_appointment_time=payload.get("return_appointment_time"),
        )
        updated_fields.extend(leg_updated)
    except ValueError as e:
        return {"error": str(e)}, 400

    if not updated_fields:
        return {"error": "Aucun champ modifié."}, 400

    change_class, severity, ack_required = classify_change(
        set(updated_fields), is_en_route=is_en_route
    )
    impact = build_operational_impact(
        change_class=change_class,
        severity=severity,
        ack_required=ack_required,
        is_en_route=is_en_route,
        changed_fields=set(updated_fields),
    )
    bump_edit_version(booking)
    after = _booking_operational_snapshot(booking)

    event = record_change_event(
        booking=booking,
        transport_request=ctx.transport_request,
        institution_id=ctx.institution_id,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_type="institution_user",
        actor_display_name=actor_display_name,
        action_type="field_updated",
        change_scope="operational",
        source="institution_portal",
        before_snapshot=before,
        after_snapshot=after,
        reason=reason or None,
        change_class=change_class,
        severity=severity,
        ack_required=ack_required,
        operational_impact=impact,
    )
    db.session.flush()

    if severity in ("WARNING", "CRITICAL"):
        fanout_critical_change(booking, event, ctx.transport_request)

    db.session.commit()
    return {
        "success": True,
        "booking_id": booking.id,
        "updated_fields": updated_fields,
        "edit_version": booking.edit_version,
        "change_event": event.serialize(),
    }, 200


def acknowledge_critical_event(
    event_id: int,
    *,
    user_id: int | None,
    actor_type: str,
    ack_channel: str,
    company_id: int | None = None,
    driver_id: int | None = None,
) -> tuple[dict[str, Any], int]:
    event = BookingChangeEvent.query.get(event_id)
    if not event:
        return {"error": "Événement introuvable"}, 404
    if not event.ack_required:
        return {"error": "Cet événement ne requiert pas d'accusé de réception"}, 400

    booking = Booking.query.get(event.booking_id)
    if not booking:
        return {"error": "Booking introuvable"}, 404

    if company_id is not None:
        cid = booking.company_id or booking.executing_company_id
        if int(cid or 0) != int(company_id):
            return {"error": "Accès refusé"}, 403
    if driver_id is not None and int(booking.driver_id or 0) != int(driver_id):
        return {"error": "Accès refusé"}, 403

    existing = BookingChangeAcknowledgement.query.filter_by(
        event_id=event_id,
        user_id=user_id,
        actor_type=actor_type,
    ).first()
    if existing:
        return {"success": True, "acknowledgement": existing.serialize()}, 200

    ack = BookingChangeAcknowledgement()
    ack.event_id = event_id
    ack.user_id = user_id
    ack.actor_type = actor_type
    ack.ack_channel = ack_channel
    db.session.add(ack)

    record_change_event(
        booking=booking,
        transport_request=None,
        institution_id=event.institution_id,
        actor_user_id=user_id,
        actor_role=None,
        actor_type=actor_type,
        actor_display_name=None,
        action_type="ack_received",
        change_scope="operational",
        source=ack_channel,
        before_snapshot=None,
        after_snapshot={"event_id": event_id},
        change_class="minor",
        severity="INFO",
        correlation_id=event.correlation_id,
    )
    db.session.commit()
    return {"success": True, "acknowledgement": ack.serialize()}, 200


def list_change_events(
    booking_id: int,
    *,
    institution_id: int | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    q = BookingChangeEvent.query.filter_by(booking_id=booking_id)
    if institution_id is not None:
        q = q.filter(
            (BookingChangeEvent.institution_id == institution_id)
            | (BookingChangeEvent.institution_id.is_(None))
        )
    rows = q.order_by(BookingChangeEvent.created_at.desc()).limit(limit).all()
    return _serialize_change_events_with_actor_names(rows)


def _serialize_change_events_with_actor_names(
    rows: list[BookingChangeEvent],
) -> list[dict[str, Any]]:
    """Sérialise les change-events en enrichissant les libellés « User #id »."""
    from shared.user_display import (
        format_user_actor_display_name,
        is_placeholder_actor_display_name,
    )

    need_ids = sorted(
        {
            int(ev.actor_user_id)
            for ev in rows
            if ev.actor_user_id
            and is_placeholder_actor_display_name(ev.actor_display_name)
        }
    )
    users_by_id: dict[int, Any] = {}
    if need_ids:
        try:
            from models.user import User

            users_by_id = {
                int(u.id): u for u in User.query.filter(User.id.in_(need_ids)).all()
            }
        except Exception as enrich_err:
            logger.warning(
                "[BookingChange] Enrichissement noms acteurs échoué: %s",
                enrich_err,
            )

    result: list[dict[str, Any]] = []
    for ev in rows:
        data = ev.serialize()
        if ev.actor_user_id and is_placeholder_actor_display_name(
            data.get("actor_display_name")
        ):
            uid = int(ev.actor_user_id)
            resolved = format_user_actor_display_name(
                user_id=uid,
                user=users_by_id.get(uid),
                fallback=data.get("actor_display_name"),
                allow_db_lookup=False,
            )
            if resolved:
                data["actor_display_name"] = resolved
        result.append(data)
    return result


def get_pending_change_request_view(booking: Any) -> dict[str, Any] | None:
    """Vue synthétique de la demande de validation active d'un booking.

    Utilisé dans booking_summary (portail institution). Retourne None si aucune
    demande active.
    """
    if booking is None:
        return None
    active_id = getattr(booking, "active_change_request_id", None)
    if not active_id:
        return None
    try:
        cr = BookingChangeRequest.query.get(active_id)
    except Exception:
        return None
    if not cr or cr.status not in TransportActionStatus.OPEN:
        return None
    return {
        "id": cr.id,
        "status": cr.status,
        "version": int(cr.version or 1),
        "action_type": cr.action_type,
        "effect_status": cr.effect_status,
        "next_actor_type": cr.next_actor_type,
        "changed_fields": cr.changed_fields,
        "proposed_patch": cr.proposed_patch,
        "before_snapshot": cr.before_snapshot,
        "after_snapshot": cr.after_snapshot,
        "reason": cr.reason,
        "requested_by_user_id": cr.requested_by_user_id,
        "requested_by_role": cr.requested_by_role,
        "expires_at": cr.expires_at.isoformat() if cr.expires_at else None,
        "created_at": cr.created_at.isoformat() if cr.created_at else None,
    }


def mask_financial_fields(data: dict[str, Any], role: str | None) -> dict[str, Any]:
    """Masque montants pour reader/requester."""
    if role in (
        InstitutionRole.ADMIN.value,
        InstitutionRole.BILLING.value,
        InstitutionRole.CURATOR.value,
    ):
        return data
    if isinstance(data, dict):
        out = dict(data)
        if "booking_summary" in out and isinstance(out["booking_summary"], dict):
            bs = dict(out["booking_summary"])
            bs.pop("amount", None)
            out["booking_summary"] = bs
        out.pop("amount", None)
        if "billing_details" in out:
            out["billing_details"] = None
        return out
    return data
