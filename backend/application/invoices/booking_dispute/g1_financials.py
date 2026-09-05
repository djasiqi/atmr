"""Oracle G1 — exactitude financière institution (lecture de l'état validé).

Ne consomme jamais une intention de correction (`proposed_*`).
Seul l'état persisté du booking entre dans les totaux.
"""

from __future__ import annotations

from typing import Any

from application.invoices.billable_amount import calculate_billable_booking_amount
from application.invoices.institution_invoice_eligibility import (
    is_institution_invoice_eligible,
    resolve_invoice_payer_type,
)
from application.invoices.institution_invoice_reconciliation import (
    classify_booking_bucket,
)


def line_financials(booking: Any, dispute: Any | None = None) -> dict[str, Any]:
    """Ligne : payeur / montant effectifs + éligibilité institution."""
    bucket, reason = classify_booking_bucket(booking)
    amount = calculate_billable_booking_amount(booking).amount_ht
    payer = resolve_invoice_payer_type(booking)
    proposed_amount = None
    proposed_payer = None
    if dispute is not None:
        raw_amt = getattr(dispute, "proposed_amount_ht", None)
        try:
            proposed_amount = float(raw_amt) if raw_amt is not None else None
        except (TypeError, ValueError):
            proposed_amount = None
        proposed_payer = getattr(dispute, "proposed_payer_type", None)
    return {
        "is_billable_to_institution": bucket == "clinic_billable",
        "effective_payer": payer,
        "effective_amount": float(amount),
        "bucket": bucket,
        "exclusion_reason": reason,
        "eligible": is_institution_invoice_eligible(booking),
        "proposed_amount_ht": proposed_amount,
        "proposed_payer_type": proposed_payer,
    }


def institution_surface(bookings: list[Any]) -> dict[str, Any]:
    """Plan institution dérivé des lignes — même règle d'éligibilité partout."""
    eligible_ids: list[int] = []
    excluded_ids: list[int] = []
    total = 0.0
    for booking in bookings:
        row = line_financials(booking)
        try:
            bid = int(booking.id)
        except (TypeError, ValueError):
            continue
        if row["is_billable_to_institution"]:
            eligible_ids.append(bid)
            total = round(total + row["effective_amount"], 2)
        else:
            excluded_ids.append(bid)
    return {
        "eligible_lines": eligible_ids,
        "excluded_lines": excluded_ids,
        "institution_total": total,
    }


def preview_institution_total(bookings: list[Any]) -> float:
    """Même filtre que la preview clinique : éligible + payeur clinic + montant canonique."""
    total = 0.0
    for booking in bookings:
        if not is_institution_invoice_eligible(booking):
            continue
        if resolve_invoice_payer_type(booking) != "clinic":
            continue
        total = round(
            total + float(calculate_billable_booking_amount(booking).amount_ht),
            2,
        )
    return total
