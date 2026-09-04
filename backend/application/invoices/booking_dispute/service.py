"""Résolution des contestations — pas de DELETE, preuves obligatoires côté transporteur."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from ext import db
from models import Booking, TransportRequest
from models.booking_dispute import (
    BookingDispute,
    BookingDisputeEvent,
    BookingDisputeEvidence,
)
from models.enums import (
    BookingDisputeStatus,
    InstitutionBillingControlStatus,
    InvoiceBillingStatus,
)
from services.institutions.booking_change_service import (
    bump_edit_version,
    record_change_event,
)

from .freeze import OPEN_DISPUTE_STATUSES, get_open_dispute_for_booking

CARRIER_STANCES = frozenset(
    {"institution_right", "mission_done", "needs_correction"}
)
EXCLUSION_REASONS = frozenset(
    {"created_by_error", "mission_cancelled", "duplicate", "other"}
)
EVIDENCE_KINDS = frozenset(
    {
        "signed_transport_sheet",
        "pickup_proof",
        "gps_history",
        "actual_times",
        "institution_written",
        "patient_confirmation",
        "other",
        "system_snapshot",
    }
)
DECIDE_ACTIONS = frozenset({"accept_carrier", "reject_evidence"})
INSTITUTION_OR_ADMIN_ROLES = frozenset(
    {
        "institution_admin",
        "institution_billing",
        "admin",
        "ADMIN",
        "platform_admin",
        "BILLING",
        "institution",
    }
)

_CARRIER_CANNOT_SELF_RESOLVE = (
    "Le transporteur ne peut pas lever lui-même la contestation. "
    "Soumettez une preuve : l'institution ou un opérateur LIRIE tranche."
)


@dataclass(frozen=True, slots=True)
class DisputeResult:
    ok: bool
    error: str | None = None
    status_code: int = 200
    dispute: BookingDispute | None = None


def _now() -> datetime:
    return datetime.now(UTC)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def get_open_dispute(booking_id: int) -> BookingDispute | None:
    return get_open_dispute_for_booking(booking_id)


def _append_event(
    dispute: BookingDispute,
    *,
    event_type: str,
    actor_user_id: int | None,
    actor_role: str | None,
    payload: dict[str, Any] | None = None,
) -> None:
    db.session.add(
        BookingDisputeEvent(
            dispute=dispute,
            event_type=event_type,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            payload=payload or {},
        )
    )


def _system_snapshot(booking: Booking) -> dict[str, Any]:
    status = getattr(booking, "status", None)
    return {
        "driver_id": getattr(booking, "driver_id", None),
        "status": str(getattr(status, "value", status) or ""),
        "scheduled_time": (
            booking.scheduled_time.isoformat()
            if getattr(booking, "scheduled_time", None)
            else None
        ),
        "completed_at": (
            booking.completed_at.isoformat()
            if getattr(booking, "completed_at", None)
            else None
        ),
        "gps_available": bool(
            getattr(booking, "pickup_lat", None) is not None
            and getattr(booking, "dropoff_lat", None) is not None
        ),
    }


def _institution_id_of(booking: Booking) -> int | None:
    resolve = getattr(booking, "_resolve_source_transport_request", None)
    req = resolve() if callable(resolve) else None
    if req is not None and getattr(req, "institution_id", None):
        try:
            return int(req.institution_id)
        except (TypeError, ValueError):
            return None
    return None


def ensure_open_dispute(
    booking: Booking,
    *,
    actor_user_id: int | None = None,
    actor_role: str | None = None,
    reason_code: str | None = None,
    reason_text: str | None = None,
) -> BookingDispute:
    """Ouvre (ou réutilise) la contestation. Ne supprime jamais le booking."""
    existing = get_open_dispute(int(booking.id))
    if existing is not None:
        return existing
    latest = (
        db.session.query(BookingDispute)
        .filter(BookingDispute.booking_id == int(booking.id))
        .order_by(BookingDispute.id.desc())
        .first()
    )
    if latest is not None and latest.status in OPEN_DISPUTE_STATUSES:
        return latest

    dispute = BookingDispute(
        booking_id=int(booking.id),
        company_id=getattr(booking, "company_id", None),
        institution_id=_institution_id_of(booking),
        status=BookingDisputeStatus.DISPUTED.value,
        opened_at=_now(),
        opened_by_user_id=actor_user_id,
        institution_reason_code=reason_code,
        institution_reason_text=reason_text
        or getattr(booking, "institution_control_anomaly_reason", None),
        frozen_amount_ht=getattr(booking, "amount", None),
        frozen_payer_type=str(getattr(booking, "billed_to_type", None) or "patient"),
        frozen_billing_party_id=getattr(booking, "billing_party_id", None),
    )
    db.session.add(dispute)
    db.session.flush()
    _append_event(
        dispute,
        event_type="opened",
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        payload={
            "reason_code": dispute.institution_reason_code,
            "frozen_amount_ht": _as_float(dispute.frozen_amount_ht),
            "frozen_payer_type": dispute.frozen_payer_type,
        },
    )
    return dispute


def present_dispute(dispute: BookingDispute, booking: Booking) -> dict[str, Any]:
    patient = getattr(booking, "customer_name", None) or ""
    ip = getattr(booking, "institution_patient", None)
    if ip is not None:
        parts = [p for p in (getattr(ip, "first_name", None), getattr(ip, "last_name", None)) if p]
        if parts:
            patient = " ".join(parts)
    scheduled = getattr(booking, "scheduled_time", None)
    history = [
        {
            "event_type": ev.event_type,
            "actor_role": ev.actor_role,
            "payload": ev.payload or {},
            "created_at": ev.created_at.isoformat() if ev.created_at else None,
        }
        for ev in (dispute.events or [])
    ]
    evidence = [
        {
            "id": int(row.id),
            "kind": row.kind,
            "source": row.source,
            "note": row.note,
            "has_file": bool(row.stored_path),
            "original_filename": row.original_filename,
            "payload": row.payload or {},
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in (dispute.evidence or [])
    ]
    return {
        "id": int(dispute.id),
        "booking_id": int(dispute.booking_id),
        "status": dispute.status,
        "treatable": dispute.status in OPEN_DISPUTE_STATUSES,
        "patient_name": patient,
        "scheduled_at": scheduled.isoformat() if scheduled else None,
        "amount_ht": _as_float(dispute.frozen_amount_ht)
        or _as_float(getattr(booking, "amount", None)),
        "institution_reason_code": dispute.institution_reason_code,
        "institution_reason_text": dispute.institution_reason_text,
        "carrier_stance": dispute.carrier_stance,
        "carrier_exclusion_reason": dispute.carrier_exclusion_reason,
        "carrier_note": dispute.carrier_note,
        "frozen": {
            "amount_ht": _as_float(dispute.frozen_amount_ht),
            "payer_type": dispute.frozen_payer_type,
        },
        "proposed_correction": {
            "amount_ht": _as_float(dispute.proposed_amount_ht),
            "payer_type": dispute.proposed_payer_type,
            "note": dispute.proposed_correction_note,
        },
        "system_facts": _system_snapshot(booking),
        "evidence": evidence,
        "history": history,
        "opened_at": dispute.opened_at.isoformat() if dispute.opened_at else None,
        "resolved_at": dispute.resolved_at.isoformat() if dispute.resolved_at else None,
        "resolver_role": dispute.resolver_role,
        "resolution_note": dispute.resolution_note,
        "invoice_billing_status": getattr(booking, "invoice_billing_status", None),
        "never_deleted": True,
    }


def confirm_institution_right(
    booking: Booking,
    *,
    exclusion_reason: str,
    note: str | None,
    actor_user_id: int | None,
    actor_role: str | None,
    actor_display_name: str | None = None,
) -> DisputeResult:
    """L'entreprise confirme : prestation non facturable, course conservée."""
    code = (exclusion_reason or "").strip()
    if code not in EXCLUSION_REASONS:
        return DisputeResult(
            ok=False,
            error=f"Motif d'exclusion invalide. Valeurs : {sorted(EXCLUSION_REASONS)}",
            status_code=400,
        )
    dispute = ensure_open_dispute(
        booking, actor_user_id=actor_user_id, actor_role=actor_role
    )
    if dispute.status not in OPEN_DISPUTE_STATUSES:
        return DisputeResult(
            ok=False,
            error="Cette contestation est déjà résolue.",
            status_code=409,
            dispute=dispute,
        )

    now = _now()
    dispute.carrier_stance = "institution_right"
    dispute.carrier_exclusion_reason = code
    dispute.carrier_note = (note or "").strip() or None
    dispute.carrier_responded_at = now
    dispute.carrier_responded_by_user_id = actor_user_id
    dispute.status = BookingDisputeStatus.RESOLVED_INSTITUTION.value
    dispute.resolved_at = now
    dispute.resolved_by_user_id = actor_user_id
    dispute.resolver_role = actor_role or "company"
    dispute.resolution_note = dispute.carrier_note
    booking.invoice_billing_status = InvoiceBillingStatus.NOT_BILLABLE.value
    bump_edit_version(booking)
    _append_event(
        dispute,
        event_type="resolved_institution",
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        payload={"exclusion_reason": code, "note": dispute.carrier_note},
    )
    _record_billing_event(
        booking,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_display_name=actor_display_name,
        action_type="dispute_resolved_institution",
        reason=f"Exclusion définitive après contestation ({code})",
    )
    return DisputeResult(ok=True, dispute=dispute)


def carrier_respond(
    booking: Booking,
    *,
    stance: str,
    note: str | None = None,
    exclusion_reason: str | None = None,
    proposed_amount_ht: float | None = None,
    proposed_payer_type: str | None = None,
    proposed_correction_note: str | None = None,
    actor_user_id: int | None = None,
    actor_role: str | None = None,
    actor_display_name: str | None = None,
) -> DisputeResult:
    raw = (stance or "").strip()
    if raw not in CARRIER_STANCES:
        return DisputeResult(
            ok=False,
            error=f"Position invalide. Valeurs : {sorted(CARRIER_STANCES)}",
            status_code=400,
        )
    if raw == "institution_right":
        return confirm_institution_right(
            booking,
            exclusion_reason=exclusion_reason or "other",
            note=note,
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            actor_display_name=actor_display_name,
        )

    dispute = ensure_open_dispute(
        booking, actor_user_id=actor_user_id, actor_role=actor_role
    )
    if dispute.status not in OPEN_DISPUTE_STATUSES:
        return DisputeResult(
            ok=False,
            error="Cette contestation est déjà résolue.",
            status_code=409,
            dispute=dispute,
        )

    now = _now()
    dispute.carrier_stance = raw
    dispute.carrier_note = (note or "").strip() or None
    dispute.carrier_responded_at = now
    dispute.carrier_responded_by_user_id = actor_user_id
    if raw == "needs_correction":
        dispute.status = BookingDisputeStatus.AWAITING_CORRECTION.value
        dispute.proposed_amount_ht = (
            Decimal(str(proposed_amount_ht)) if proposed_amount_ht is not None else None
        )
        dispute.proposed_payer_type = (proposed_payer_type or "").strip() or None
        dispute.proposed_correction_note = (proposed_correction_note or note or "").strip() or None
    else:
        dispute.status = BookingDisputeStatus.AWAITING_CARRIER_RESPONSE.value
        _ensure_system_snapshot(dispute, booking, actor_user_id)

    _append_event(
        dispute,
        event_type="carrier_responded",
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        payload={"stance": raw},
    )
    return DisputeResult(ok=True, dispute=dispute)


def _ensure_system_snapshot(
    dispute: BookingDispute, booking: Booking, actor_user_id: int | None
) -> None:
    already = any(row.kind == "system_snapshot" for row in (dispute.evidence or []))
    if already:
        return
    db.session.add(
        BookingDisputeEvidence(
            dispute=dispute,
            kind="system_snapshot",
            source="system",
            note="Éléments déjà enregistrés par le système (pas une preuve documentaire).",
            payload=_system_snapshot(booking),
            created_by_user_id=actor_user_id,
        )
    )


def add_carrier_evidence(
    booking: Booking,
    *,
    kind: str,
    note: str | None = None,
    stored_path: str | None = None,
    original_filename: str | None = None,
    actor_user_id: int | None = None,
    actor_role: str | None = None,
) -> DisputeResult:
    code = (kind or "").strip()
    if code not in EVIDENCE_KINDS or code == "system_snapshot":
        return DisputeResult(
            ok=False,
            error=f"Type de preuve invalide. Valeurs : {sorted(EVIDENCE_KINDS - {'system_snapshot'})}",
            status_code=400,
        )
    dispute = get_open_dispute(int(booking.id))
    if dispute is None:
        return DisputeResult(
            ok=False,
            error="Aucune contestation ouverte.",
            status_code=404,
        )
    if dispute.status not in (
        BookingDisputeStatus.AWAITING_CARRIER_RESPONSE.value,
        BookingDisputeStatus.AWAITING_CORRECTION.value,
        BookingDisputeStatus.DISPUTED.value,
        BookingDisputeStatus.EVIDENCE_SUBMITTED.value,
    ):
        return DisputeResult(
            ok=False,
            error="Impossible d'ajouter une preuve sur une contestation close.",
            status_code=409,
            dispute=dispute,
        )
    if dispute.status == BookingDisputeStatus.DISPUTED.value:
        dispute.status = BookingDisputeStatus.AWAITING_CARRIER_RESPONSE.value
        if not dispute.carrier_stance:
            dispute.carrier_stance = "mission_done"
    db.session.add(
        BookingDisputeEvidence(
            dispute=dispute,
            kind=code,
            source="uploaded",
            note=(note or "").strip() or None,
            stored_path=stored_path,
            original_filename=original_filename,
            created_by_user_id=actor_user_id,
        )
    )
    _append_event(
        dispute,
        event_type="evidence_added",
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        payload={"kind": code},
    )
    return DisputeResult(ok=True, dispute=dispute)


def submit_dispute_for_validation(
    booking: Booking,
    *,
    actor_user_id: int | None = None,
    actor_role: str | None = None,
) -> DisputeResult:
    dispute = get_open_dispute(int(booking.id))
    if dispute is None:
        return DisputeResult(ok=False, error="Aucune contestation ouverte.", status_code=404)
    if dispute.carrier_stance not in ("mission_done", "needs_correction"):
        return DisputeResult(
            ok=False,
            error="Choisissez d'abord « mission effectuée » ou « informations à corriger ».",
            status_code=400,
            dispute=dispute,
        )
    human_proofs = [
        row
        for row in (dispute.evidence or [])
        if row.source == "uploaded"
    ]
    if not human_proofs:
        return DisputeResult(
            ok=False,
            error="Ajoutez au moins un justificatif (le snapshot système ne suffit pas).",
            status_code=400,
            dispute=dispute,
        )
    dispute.status = BookingDisputeStatus.EVIDENCE_SUBMITTED.value
    dispute.submitted_at = _now()
    _append_event(
        dispute,
        event_type="submitted_for_validation",
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        payload={"evidence_count": len(human_proofs)},
    )
    return DisputeResult(ok=True, dispute=dispute)


def _actor_can_decide(actor_role: str | None) -> bool:
    raw = str(actor_role or "").strip()
    return raw in INSTITUTION_OR_ADMIN_ROLES or raw.lower() in {
        "institution_admin",
        "institution_billing",
        "platform_admin",
        "admin",
    }


def decide_dispute(
    booking: Booking,
    *,
    decision: str,
    note: str | None,
    actor_user_id: int | None,
    actor_role: str | None,
    actor_display_name: str | None = None,
) -> DisputeResult:
    if not _actor_can_decide(actor_role):
        return DisputeResult(
            ok=False,
            error=_CARRIER_CANNOT_SELF_RESOLVE,
            status_code=403,
        )
    action = (decision or "").strip()
    if action not in DECIDE_ACTIONS:
        return DisputeResult(
            ok=False,
            error=f"Décision invalide. Valeurs : {sorted(DECIDE_ACTIONS)}",
            status_code=400,
        )
    dispute = get_open_dispute(int(booking.id))
    if dispute is None:
        return DisputeResult(ok=False, error="Aucune contestation ouverte.", status_code=404)
    if dispute.status != BookingDisputeStatus.EVIDENCE_SUBMITTED.value:
        return DisputeResult(
            ok=False,
            error="La preuve n'a pas encore été soumise pour validation.",
            status_code=409,
            dispute=dispute,
        )

    if action == "reject_evidence":
        dispute.status = BookingDisputeStatus.AWAITING_CARRIER_RESPONSE.value
        dispute.resolution_note = (note or "").strip() or None
        _append_event(
            dispute,
            event_type="evidence_rejected",
            actor_user_id=actor_user_id,
            actor_role=actor_role,
            payload={"note": dispute.resolution_note},
        )
        return DisputeResult(ok=True, dispute=dispute)

    now = _now()
    if dispute.carrier_stance == "needs_correction":
        _apply_proposed_correction(booking, dispute)
    dispute.status = BookingDisputeStatus.RESOLVED_CARRIER.value
    dispute.resolved_at = now
    dispute.resolved_by_user_id = actor_user_id
    dispute.resolver_role = actor_role or "institution"
    dispute.resolution_note = (note or "").strip() or None
    booking.invoice_billing_status = InvoiceBillingStatus.BILLABLE.value
    booking.institution_control_status = InstitutionBillingControlStatus.VALIDATED
    booking.institution_control_validated_at = now
    booking.institution_control_validated_by_user_id = actor_user_id
    booking.institution_control_validated_by_display_name = (
        actor_display_name or ""
    ).strip() or None
    booking.institution_control_anomaly_reason = None
    bump_edit_version(booking)
    _append_event(
        dispute,
        event_type="resolved_carrier",
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        payload={"note": dispute.resolution_note},
    )
    _record_billing_event(
        booking,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_display_name=actor_display_name,
        action_type="dispute_resolved_carrier",
        reason="Mission confirmée après contestation — justificatif validé",
    )
    return DisputeResult(ok=True, dispute=dispute)


def _apply_proposed_correction(booking: Booking, dispute: BookingDispute) -> None:
    if dispute.proposed_amount_ht is not None:
        booking.amount = float(dispute.proposed_amount_ht)
    if dispute.proposed_payer_type:
        booking.billed_to_type = dispute.proposed_payer_type


def _record_billing_event(
    booking: Booking,
    *,
    actor_user_id: int | None,
    actor_role: str | None,
    actor_display_name: str | None,
    action_type: str,
    reason: str,
) -> None:
    resolve = getattr(booking, "_resolve_source_transport_request", None)
    req: TransportRequest | None = resolve() if callable(resolve) else None
    if req is None:
        return
    record_change_event(
        booking=booking,
        transport_request=req,
        institution_id=getattr(req, "institution_id", None),
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_type="company_user" if (actor_role or "").upper() in {"COMPANY", "ADMIN"} else "institution_user",
        actor_display_name=actor_display_name,
        action_type=action_type,
        change_scope="billing_control",
        source="dispute_resolution",
        before_snapshot=None,
        after_snapshot={"invoice_billing_status": getattr(booking, "invoice_billing_status", None)},
        reason=reason,
        change_class="major",
        severity="INFO",
        financial_actor_role="billing",
    )


def latest_dispute_summaries(booking_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not booking_ids:
        return {}
    rows = (
        db.session.query(BookingDispute)
        .filter(BookingDispute.booking_id.in_(booking_ids))
        .order_by(BookingDispute.booking_id, BookingDispute.id.desc())
        .all()
    )
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        bid = int(row.booking_id)
        if bid in out:
            continue
        out[bid] = {
            "dispute_id": int(row.id),
            "dispute_status": row.status,
            "dispute_treatable": row.status in OPEN_DISPUTE_STATUSES,
        }
    return out
