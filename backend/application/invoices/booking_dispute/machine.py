"""Contrat G2 — états explicites de contestation (lecture, pas de nouveau produit)."""

from __future__ import annotations

from typing import Any

from application.invoices.institution_invoice_reconciliation import (
    classify_booking_bucket,
)
from models.enums import BookingDisputeStatus

from .freeze import OPEN_DISPUTE_STATUSES

TERMINAL_STATUSES = frozenset(
    {
        BookingDisputeStatus.RESOLVED_INSTITUTION.value,
        BookingDisputeStatus.RESOLVED_CARRIER.value,
    }
)

CARRIER_ACTABLE_STATUSES = frozenset(
    {
        BookingDisputeStatus.DISPUTED.value,
        BookingDisputeStatus.AWAITING_CARRIER_RESPONSE.value,
        BookingDisputeStatus.AWAITING_CORRECTION.value,
    }
)

THIRD_PARTY_ACTABLE_STATUSES = frozenset(
    {BookingDisputeStatus.EVIDENCE_SUBMITTED.value}
)

ALLOWED_CORRECTION_PAYERS = frozenset({"clinic", "patient"})

_SUBMITTABLE_STATUSES = frozenset(
    {
        BookingDisputeStatus.AWAITING_CARRIER_RESPONSE.value,
        BookingDisputeStatus.AWAITING_CORRECTION.value,
    }
)


def who_may_act(status: str | None) -> tuple[str, ...]:
    raw = str(status or "")
    if raw in THIRD_PARTY_ACTABLE_STATUSES:
        return ("institution", "admin")
    if raw in CARRIER_ACTABLE_STATUSES:
        return ("COMPANY",)
    return ()


def clinic_line_in_invoice(booking: Any) -> bool:
    bucket, _reason = classify_booking_bucket(booking)
    return bucket == "clinic_billable"


def snapshot(booking: Any, dispute: Any) -> dict[str, Any]:
    """État déterministe d'une ligne contestée — contrat G2."""
    bucket, reason = classify_booking_bucket(booking)
    status = str(getattr(dispute, "status", "") or "")
    proposed = getattr(dispute, "proposed_amount_ht", None)
    try:
        proposed_amount = float(proposed) if proposed is not None else None
    except (TypeError, ValueError):
        proposed_amount = None
    amount = getattr(booking, "amount", None)
    try:
        amount_ht = float(amount) if amount is not None else None
    except (TypeError, ValueError):
        amount_ht = None
    return {
        "status": status,
        "stance": getattr(dispute, "carrier_stance", None),
        "open": status in OPEN_DISPUTE_STATUSES,
        "terminal": status in TERMINAL_STATUSES,
        "who_may_act": who_may_act(status),
        "carrier_can_close": False,
        "bucket": bucket,
        "exclusion_reason": reason,
        "clinic_line_in_invoice": bucket == "clinic_billable",
        "payer": str(getattr(booking, "billed_to_type", None) or ""),
        "amount_ht": amount_ht,
        "proposed_amount_ht": proposed_amount,
        "proposed_payer_type": getattr(dispute, "proposed_payer_type", None),
        "invoice_billing_status": getattr(booking, "invoice_billing_status", None),
    }


def is_carrier_actable(status: str | None) -> bool:
    return str(status or "") in CARRIER_ACTABLE_STATUSES


def is_submittable(status: str | None) -> bool:
    return str(status or "") in _SUBMITTABLE_STATUSES
