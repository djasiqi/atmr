"""Contrôle institution pré-facturation — statuts et helpers."""

from __future__ import annotations

from typing import Any

from models import Booking
from models.enums import InstitutionBillingControlStatus


def persisted_control_status(booking: Booking) -> str | None:
    raw = getattr(booking, "institution_control_status", None)
    if raw is None:
        return None
    return str(getattr(raw, "value", raw))


def effective_control_status(booking: Booking) -> str:
    """Statut effectif : NULL legacy → ``pending_review`` sans write."""
    persisted = persisted_control_status(booking)
    if persisted is None:
        return InstitutionBillingControlStatus.PENDING_REVIEW.value
    return persisted


def control_status_snapshot(booking: Booking) -> dict[str, Any]:
    return {
        "control_status": effective_control_status(booking),
        "persisted_control_status": persisted_control_status(booking),
        "validated_at": (
            booking.institution_control_validated_at.isoformat()
            if getattr(booking, "institution_control_validated_at", None)
            else None
        ),
        "validated_by_user_id": getattr(
            booking, "institution_control_validated_by_user_id", None
        ),
        "validated_by_display_name": getattr(
            booking, "institution_control_validated_by_display_name", None
        ),
        "anomaly_reason": getattr(booking, "institution_control_anomaly_reason", None),
    }


def reset_control_after_payer_correction(booking: Booking) -> None:
    """Correction payeur : repasse en ``pending_review`` — jamais auto-validé."""
    booking.institution_control_status = InstitutionBillingControlStatus.PENDING_REVIEW
    booking.institution_control_validated_at = None
    booking.institution_control_validated_by_user_id = None
    booking.institution_control_validated_by_display_name = None
    booking.institution_control_anomaly_reason = None


def clear_validation_fields(booking: Booking) -> None:
    booking.institution_control_validated_at = None
    booking.institution_control_validated_by_user_id = None
    booking.institution_control_validated_by_display_name = None
