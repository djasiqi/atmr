"""Gel financier pendant une contestation ouverte."""

from __future__ import annotations

from typing import Any

from models.booking_dispute import BookingDispute
from models.enums import BookingDisputeStatus, InstitutionBillingControlStatus

OPEN_DISPUTE_STATUSES = frozenset(
    {
        BookingDisputeStatus.DISPUTED.value,
        BookingDisputeStatus.AWAITING_CARRIER_RESPONSE.value,
        BookingDisputeStatus.EVIDENCE_SUBMITTED.value,
        BookingDisputeStatus.AWAITING_CORRECTION.value,
    }
)

_FREEZE_MESSAGE = (
    "Contestation en cours : montant et payeur sont gelés jusqu'à la résolution. "
    "Toute correction doit passer par le traitement de la contestation."
)


def is_open_dispute_status(status: str | None) -> bool:
    return str(status or "") in OPEN_DISPUTE_STATUSES


def get_open_dispute_for_booking(booking_id: int) -> BookingDispute | None:
    from ext import db

    return (
        db.session.query(BookingDispute)
        .filter(
            BookingDispute.booking_id == int(booking_id),
            BookingDispute.status.in_(sorted(OPEN_DISPUTE_STATUSES)),
        )
        .order_by(BookingDispute.id.desc())
        .first()
    )


def financial_change_blocked_by_dispute(booking: Any) -> tuple[bool, str | None]:
    """Bloque un changement silencieux de montant / payeur pendant le litige."""
    bid = getattr(booking, "id", None)
    if bid is None:
        return False, None
    if get_open_dispute_for_booking(int(bid)) is not None:
        return True, _FREEZE_MESSAGE
    raw = getattr(booking, "institution_control_status", None)
    persisted = str(getattr(raw, "value", raw) or "")
    if persisted == InstitutionBillingControlStatus.ANOMALY.value:
        billing = str(getattr(booking, "invoice_billing_status", None) or "")
        if billing != "not_billable":
            return True, _FREEZE_MESSAGE
    return False, None
