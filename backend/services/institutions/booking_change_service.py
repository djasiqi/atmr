"""Service métier : audit trail, versioning et alertes critiques institution."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from application.bookings.cancellation_rules import (
    compute_cancellation_fee,
    get_cancellation_display_label,
)
from ext import db
from models import Booking, BookingChangeAcknowledgement, BookingChangeEvent, TransportRequest
from models.enums import BookingStatus, InstitutionRole
from shared.time_utils import parse_local_naive

logger = logging.getLogger(__name__)

MIN_CRITICAL_REASON_LEN = 10

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
    return {k: before.get(k) != after.get(k) for k in keys if before.get(k) != after.get(k)}


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
            & {"pickup_location", "dropoff_location", "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon"}
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
        return f"Rôle requis: {', '.join(sorted(OPERATIONAL_ROLES))}. Votre rôle: {role}"
    return None


def assert_not_boarded(booking: Booking) -> str | None:
    if booking.boarded_at is not None:
        return (
            "Modification impossible : le patient est déjà pris en charge "
            f"(boarded_at={booking.boarded_at.isoformat()})."
        )
    return None


def check_version(booking: Booking, client_version: int | None) -> dict[str, Any] | None:
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
    return event


def bump_edit_version(booking: Booking) -> int:
    booking.edit_version = int(booking.edit_version or 1) + 1
    booking.updated_at = datetime.now(UTC)
    return booking.edit_version


def apply_operational_patch(
    booking: Booking, validated: dict[str, Any]
) -> list[str]:
    updated: list[str] = []
    for key, value in validated.items():
        if key not in INSTITUTION_OPERATIONAL_FIELDS:
            continue
        if key == "scheduled_time":
            scheduled_local = parse_local_naive(value)
            if scheduled_local is None:
                raise ValueError("Heure planifiée invalide.")
            booking.scheduled_time = scheduled_local
            if hasattr(booking, "time_confirmed"):
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
    return updated


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
    public_id = getattr(transport_request, "public_id", None) if transport_request else None
    patient = booking.customer_name or "Patient"
    msg = (
        f"Modification institution en route — {patient} "
        f"(course #{booking.id})"
    )
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

    is_en_route = status == BookingStatus.EN_ROUTE.value
    if is_en_route and len((reason or "").strip()) < MIN_CRITICAL_REASON_LEN:
        return {
            "error": f"Motif obligatoire (min. {MIN_CRITICAL_REASON_LEN} caractères) pour annulation en route.",
        }, 400

    before = _booking_operational_snapshot(booking)
    cancelled_at = datetime.now(UTC)
    fee = compute_cancellation_fee(
        booking,
        status_at_cancel=status,
        cancelled_at=cancelled_at,
        reason_code=reason_code or "CLIENT_REQUEST",
    )
    is_billable = True if is_en_route else fee.is_billable

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

    db.session.commit()
    return {
        "success": True,
        "booking_id": booking.id,
        "status": _status_value(booking.status),
        "is_cancellation_billable": is_billable,
        "edit_version": booking.edit_version,
        "change_event_id": event.id,
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
    unknown = set(payload.keys()) - INSTITUTION_OPERATIONAL_FIELDS - {"version", "reason"}
    if unknown:
        return {
            "error": "Champs non autorisés.",
            "rejected_fields": sorted(unknown),
        }, 400

    patch = {k: v for k, v in payload.items() if k in INSTITUTION_OPERATIONAL_FIELDS}
    if not patch:
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
    _cc, _sev, ack_required = classify_change(
        changed_preview, is_en_route=is_en_route
    )
    if is_en_route and (changed_preview & CRITICAL_EN_ROUTE_FIELDS):
        if len(reason) < MIN_CRITICAL_REASON_LEN:
            return {
                "error": f"Motif obligatoire (min. {MIN_CRITICAL_REASON_LEN} caractères) pour modification en route.",
            }, 400

    before = _booking_operational_snapshot(booking)
    try:
        updated_fields = apply_operational_patch(booking, patch)
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
    rows = (
        q.order_by(BookingChangeEvent.created_at.desc()).limit(limit).all()
    )
    return [r.serialize() for r in rows]


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
