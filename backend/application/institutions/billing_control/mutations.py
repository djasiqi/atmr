"""Mutations workflow contrôle institution — validate / anomaly / reopen."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from application.companies.reservations.billing_adjustment import (
    booking_billing_is_locked,
)
from application.institutions.billing_control.status import (
    clear_validation_fields,
    control_status_snapshot,
    effective_control_status,
)
from models import Booking, TransportRequest
from models.enums import InstitutionBillingControlStatus
from services.institutions.booking_change_service import (
    bump_edit_version,
    record_change_event,
)

ANOMALY_REASON_CODES = frozenset(
    {
        "PAYER_NOT_FOUND",
        "TRANSPORT_DISPUTED",
        "FINANCIAL_INCONSISTENCY",
        "MISSING_BLOCKING_DATA",
        "OTHER",
    }
)


@dataclass(frozen=True, slots=True)
class ControlMutationResult:
    ok: bool
    error: str | None = None
    status_code: int = 200
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    audit_event_id: int | None = None


def control_state_is_readonly(booking: Booking) -> tuple[bool, str | None]:
    locked, msg = booking_billing_is_locked(booking)
    if locked:
        return True, msg or "Booking facturé ou verrouillé : contrôle en lecture seule."
    return False, None


def validate_booking_control(
    booking: Booking,
    *,
    transport_request: TransportRequest,
    institution_id: int,
    actor_user_id: int | None,
    actor_role: str | None,
    actor_display_name: str | None,
) -> ControlMutationResult:
    readonly, msg = control_state_is_readonly(booking)
    if readonly:
        return ControlMutationResult(ok=False, error=msg, status_code=409)

    effective = effective_control_status(booking)
    if effective == InstitutionBillingControlStatus.VALIDATED.value:
        return ControlMutationResult(
            ok=False,
            error="Booking déjà validé.",
            status_code=409,
        )
    if effective == InstitutionBillingControlStatus.ANOMALY.value:
        return ControlMutationResult(
            ok=False,
            error="Résoudre l'anomalie avant validation (reopen puis corriger).",
            status_code=409,
        )

    before = control_status_snapshot(booking)
    now = datetime.now(UTC)
    booking.institution_control_status = InstitutionBillingControlStatus.VALIDATED
    booking.institution_control_validated_at = now
    booking.institution_control_validated_by_user_id = actor_user_id
    booking.institution_control_validated_by_display_name = (
        actor_display_name or ""
    ).strip() or None
    booking.institution_control_anomaly_reason = None
    bump_edit_version(booking)
    after = control_status_snapshot(booking)

    event = record_change_event(
        booking=booking,
        transport_request=transport_request,
        institution_id=institution_id,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_type="institution_user",
        actor_display_name=actor_display_name,
        action_type="billing_control_validated",
        change_scope="billing_control",
        source="institution_portal",
        before_snapshot=before,
        after_snapshot=after,
        reason="Validation contrôle facturation institution",
        change_class="major",
        severity="INFO",
        financial_actor_role=(
            "billing" if actor_role == "institution_billing" else "admin"
        ),
    )
    return ControlMutationResult(
        ok=True,
        before=before,
        after=after,
        audit_event_id=int(event.id) if getattr(event, "id", None) else None,
    )


def mark_booking_control_anomaly(
    booking: Booking,
    *,
    transport_request: TransportRequest,
    institution_id: int,
    actor_user_id: int | None,
    actor_role: str | None,
    actor_display_name: str | None,
    anomaly_reason_code: str,
    anomaly_comment: str | None = None,
) -> ControlMutationResult:
    readonly, msg = control_state_is_readonly(booking)
    if readonly:
        return ControlMutationResult(ok=False, error=msg, status_code=409)

    code = (anomaly_reason_code or "").upper().strip()
    if code not in ANOMALY_REASON_CODES:
        return ControlMutationResult(
            ok=False,
            error=f"anomaly_reason_code invalide. Valeurs: {sorted(ANOMALY_REASON_CODES)}",
            status_code=400,
        )

    reason_text = code
    if anomaly_comment and anomaly_comment.strip():
        reason_text = f"{code}: {anomaly_comment.strip()}"

    before = control_status_snapshot(booking)
    booking.institution_control_status = InstitutionBillingControlStatus.ANOMALY
    booking.institution_control_anomaly_reason = reason_text
    clear_validation_fields(booking)
    bump_edit_version(booking)
    from application.invoices.booking_dispute.service import ensure_open_dispute

    ensure_open_dispute(
        booking,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        reason_code=code,
        reason_text=reason_text,
    )
    after = control_status_snapshot(booking)

    event = record_change_event(
        booking=booking,
        transport_request=transport_request,
        institution_id=institution_id,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_type="institution_user",
        actor_display_name=actor_display_name,
        action_type="billing_control_anomaly",
        change_scope="billing_control",
        source="institution_portal",
        before_snapshot=before,
        after_snapshot=after,
        reason=reason_text,
        change_class="major",
        severity="WARNING",
        financial_actor_role=(
            "billing" if actor_role == "institution_billing" else "admin"
        ),
        billing_change_reason_code=code,
    )
    return ControlMutationResult(
        ok=True,
        before=before,
        after=after,
        audit_event_id=int(event.id) if getattr(event, "id", None) else None,
    )


def reopen_booking_control(
    booking: Booking,
    *,
    transport_request: TransportRequest,
    institution_id: int,
    actor_user_id: int | None,
    actor_role: str | None,
    actor_display_name: str | None,
    reason: str | None = None,
) -> ControlMutationResult:
    """Validé ou anomalie → pending_review (clear validation / anomaly)."""
    readonly, msg = control_state_is_readonly(booking)
    if readonly:
        return ControlMutationResult(ok=False, error=msg, status_code=409)

    current = effective_control_status(booking)
    if current not in (
        InstitutionBillingControlStatus.ANOMALY.value,
        InstitutionBillingControlStatus.VALIDATED.value,
    ):
        return ControlMutationResult(
            ok=False,
            error="Seul un booking validé ou en anomalie peut être réouvert.",
            status_code=409,
        )
    from application.invoices.booking_dispute.freeze import get_open_dispute_for_booking

    if get_open_dispute_for_booking(int(booking.id)) is not None:
        return ControlMutationResult(
            ok=False,
            error=(
                "Une contestation est en cours. Validez ou refusez le justificatif "
                "plutôt que de réouvrir silencieusement."
            ),
            status_code=409,
        )
    if str(getattr(booking, "invoice_billing_status", None) or "") == "not_billable":
        return ControlMutationResult(
            ok=False,
            error=(
                "Prestation exclue définitivement après contestation. "
                "La course reste historisée."
            ),
            status_code=409,
        )

    before = control_status_snapshot(booking)
    booking.institution_control_status = InstitutionBillingControlStatus.PENDING_REVIEW
    booking.institution_control_anomaly_reason = None
    clear_validation_fields(booking)
    bump_edit_version(booking)
    after = control_status_snapshot(booking)

    event = record_change_event(
        booking=booking,
        transport_request=transport_request,
        institution_id=institution_id,
        actor_user_id=actor_user_id,
        actor_role=actor_role,
        actor_type="institution_user",
        actor_display_name=actor_display_name,
        action_type="billing_control_reopened",
        change_scope="billing_control",
        source="institution_portal",
        before_snapshot=before,
        after_snapshot=after,
        reason=reason or "Réouverture contrôle facturation",
        change_class="major",
        severity="INFO",
        financial_actor_role=(
            "billing" if actor_role == "institution_billing" else "admin"
        ),
    )
    return ControlMutationResult(
        ok=True,
        before=before,
        after=after,
        audit_event_id=int(event.id) if getattr(event, "id", None) else None,
    )
