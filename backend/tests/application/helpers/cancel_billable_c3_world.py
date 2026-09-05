"""Monde C3 — libellés d'annulation. Tests first, pas de correctif, pas de C4."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

from application.bookings.cancellation_rules import get_cancellation_display_label
from application.invoices.generate_clinic_monthly_invoice import (
    GenerateClinicMonthlyInvoiceInput,
    GenerateClinicMonthlyInvoiceUseCase,
)
from application.invoices.period_invoice_preview import build_period_invoice_preview
from models import InvoiceLine
from tests.application.helpers.cancel_billable_c1_world import (
    FEE_HT,
    PERIOD_MONTH,
    PERIOD_YEAR,
    add_canceled_booking,
    build_c1_world,
)

HISTORICAL_FALLBACK = "Annulation (historique)"


def build_c3_world(db) -> dict[str, Any]:
    return build_c1_world(db)


def canonical_cancellation_label(
    *,
    reason_code: str | None,
    reason_text: str | None,
    persisted_label: str | None,
) -> str:
    """Priorité C3 : persisté → helper métier → historique."""
    persisted = (persisted_label or "").strip()
    if persisted:
        return persisted
    return get_cancellation_display_label(reason_code, reason_text)


def add_canceled_labeled_booking(
    db,
    world: dict[str, Any],
    *,
    reason_code: str | None,
    reason_text: str | None = None,
    persist_display_label: bool = True,
    fee_percent: int | None = None,
    fee_tier_id: str | None = None,
    fee_amount: Decimal | None = FEE_HT,
    day: int = 12,
):
    booking = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
        cancellation_fee_amount=fee_amount,
        day=day,
    )
    booking.cancellation_reason_code = reason_code
    booking.cancellation_reason_text = reason_text
    if persist_display_label:
        booking.cancellation_display_label = get_cancellation_display_label(
            reason_code, reason_text
        )
    else:
        booking.cancellation_display_label = None
    booking.cancellation_fee_percent = fee_percent
    booking.cancellation_fee_tier_id = fee_tier_id
    db.session.flush()
    return booking


def preview_clinic_description(world: dict[str, Any], booking_id: int) -> str | None:
    preview = build_period_invoice_preview(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        include_line_details=True,
    )
    for line in preview.preview_lines:
        if int(line.booking_id) == int(booking_id):
            return str(line.description or "")
    return None


def generate_clinic_description(world: dict[str, Any], booking_id: int) -> str | None:
    pdf = MagicMock()
    pdf.generate_invoice_pdf.return_value = "https://cdn.example/invoice.pdf"
    result = GenerateClinicMonthlyInvoiceUseCase(pdf_service=pdf).execute(
        GenerateClinicMonthlyInvoiceInput(
            company_id=world["transport"].id,
            clinic_company_id=world["clinic"].id,
            period_year=PERIOD_YEAR,
            period_month=PERIOD_MONTH,
        )
    )
    if not result.success or result.invoice_id is None:
        return None
    bid = int(booking_id)
    for line in InvoiceLine.query.filter_by(invoice_id=result.invoice_id).all():
        meta = line.line_meta if isinstance(line.line_meta, dict) else {}
        claimed = {int(i) for i in (meta.get("booking_ids") or [])}
        if bid in claimed or (
            line.reservation_id is not None and int(line.reservation_id) == bid
        ):
            return str(line.description or "")
    return None
