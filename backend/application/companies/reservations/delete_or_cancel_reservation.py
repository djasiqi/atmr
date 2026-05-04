from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from ._status import set_status, status_value


class _BookingLike(Protocol):
    id: int | None
    status: Any
    scheduled_time: datetime | None
    driver_id: Any
    cancelled_at: Any
    cancelled_by_role: Any
    cancellation_reason_code: Any
    cancellation_reason_text: Any
    is_cancellation_billable: Any
    cancellation_display_label: Any


@dataclass(frozen=True, slots=True)
class DeleteOrCancelCompanyReservationResult:
    ok: bool
    action: str | None = None  # "delete" | "cancel"
    message: str | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None
    should_trigger_dispatch: bool = False
    trigger_reason: str | None = None
    should_delete_assignments: bool = False
    is_cancellation_billable: bool | None = None
    cancellation_display_label: str | None = None


class DeleteOrCancelCompanyReservationUseCase:
    """Use-case Application: suppression/annulation intelligente selon
    statut + timing."""

    def _validate_cancellation_reason(
        self,
        reason_code: str | None,
        reason_text: str | None,
    ) -> DeleteOrCancelCompanyReservationResult | None:
        """Valide que le motif est present. Retourne un result d'erreur ou None si OK."""
        if not reason_code or not str(reason_code).strip():
            return DeleteOrCancelCompanyReservationResult(
                ok=False,
                error={"error": "Motif d'annulation requis (reason_code)."},
                status_code=400,
            )
        code = str(reason_code).strip().upper()
        if code == "OTHER" and (not reason_text or not str(reason_text).strip()):
            return DeleteOrCancelCompanyReservationResult(
                ok=False,
                error={
                    "error": "Justification requise pour le motif 'Autre' (reason_text)."
                },
                status_code=400,
            )
        return None

    def _cancel_with_reason(
        self,
        booking: _BookingLike,
        reason_code: str | None,
        reason_text: str | None,
        now: datetime,
        *,
        message: str = "La réservation a été annulée.",
    ) -> DeleteOrCancelCompanyReservationResult:
        """Valide le motif, applique l'annulation et persiste les champs."""
        validation_error = self._validate_cancellation_reason(reason_code, reason_text)
        if validation_error:
            return validation_error

        raw_status = getattr(booking, "status", None)
        if raw_status is not None and hasattr(raw_status, "value"):
            status_at_cancel = raw_status.value
        else:
            status_at_cancel = raw_status

        set_status(booking, "status", "CANCELED")
        if getattr(booking, "driver_id", None):
            booking.driver_id = None

        from application.bookings.cancellation_rules import (
            compute_cancellation_fields,
            log_cancellation_persisted,
        )
        from models.invoice import CompanyBillingSettings

        billing = CompanyBillingSettings.query.filter_by(
            company_id=booking.company_id
        ).first() if getattr(booking, "company_id", None) else None
        cancellation_policy = getattr(billing, "cancellation_policy", None) if billing else None

        already_had_reason = bool(getattr(booking, "cancellation_reason_code", None))
        fields = compute_cancellation_fields(
            reason_code=reason_code,
            reason_text=reason_text,
            cancelled_by_role="company",
            now=now,
            booking=booking,
            policy=cancellation_policy,
            status_at_cancel=status_at_cancel,
        )
        for key, val in fields.items():
            if hasattr(booking, key):
                setattr(booking, key, val)
        if not already_had_reason:
            log_cancellation_persisted(booking, fields)

        return DeleteOrCancelCompanyReservationResult(
            ok=True,
            action="cancel",
            message=message,
            should_trigger_dispatch=True,
            trigger_reason="cancel",
            is_cancellation_billable=fields["is_cancellation_billable"],
            cancellation_display_label=fields["cancellation_display_label"],
        )

    def execute(
        self,
        booking: _BookingLike,
        *,
        now_utc: datetime | None = None,
        hours_offset: float = -24.0,
        reason_code: str | None = None,
        reason_text: str | None = None,
    ) -> DeleteOrCancelCompanyReservationResult:
        now = now_utc or datetime.now(UTC)

        scheduled = getattr(booking, "scheduled_time", None)
        if scheduled and scheduled.tzinfo is None:
            # Si naïf: on le traite comme UTC pour l'algorithme
            # (la route peut normaliser si besoin)
            scheduled = scheduled.replace(tzinfo=UTC)

        time_diff_hours = (
            ((scheduled - now).total_seconds() / 3600.0) if scheduled else 0.0
        )

        st = status_value(getattr(booking, "status", None)).lower()

        # Règle 1: PENDING/ACCEPTED → delete (physique)
        if st in {"pending", "accepted"}:
            return DeleteOrCancelCompanyReservationResult(
                ok=True,
                action="delete",
                message="La réservation a été supprimée avec succès.",
                should_trigger_dispatch=True,
                trigger_reason="cancel",
            )

        # Règle 2: ASSIGNED → timing
        if st == "assigned":
            if time_diff_hours < hours_offset:
                return DeleteOrCancelCompanyReservationResult(
                    ok=True,
                    action="delete",
                    message="La réservation a été supprimée avec succès.",
                    should_delete_assignments=True,
                    should_trigger_dispatch=True,
                    trigger_reason="cancel",
                )
            # Validation motif obligatoire pour cancel
            return self._cancel_with_reason(
                booking, reason_code, reason_text, now,
                message="La réservation a été annulée avec succès.",
            )

        # Règle 2.5: EN_ROUTE → cancel (facturation selon motif)
        if st == "en_route":
            return self._cancel_with_reason(
                booking, reason_code, reason_text, now,
                message="La course a été annulée (chauffeur en route).",
            )

        # Règle 3: forbid
        msg_map = {
            "in_progress": "La course est en cours et ne peut pas être annulée.",
            "completed": "La course est terminée et ne peut pas être modifiée.",
            "canceled": "La course est déjà annulée.",
            "return_completed": "La course est terminée et ne peut pas être modifiée.",
        }
        msg = msg_map.get(st) or (
            "Impossible de supprimer/annuler une course avec le statut "
            + f"'{status_value(getattr(booking, 'status', None))}'."
        )
        return DeleteOrCancelCompanyReservationResult(
            ok=False,
            error={"error": msg},
            status_code=403,
        )
