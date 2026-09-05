"""Monde C2 — montant d'annulation. Pas de correctif, pas de C3/C4."""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

from application.invoices.billable_amount import (
    BillableAmount,
    calculate_billable_booking_amount,
)
from application.invoices.billing_opportunities import list_billing_opportunities
from application.invoices.generate_clinic_monthly_invoice import (
    GenerateClinicMonthlyInvoiceInput,
    GenerateClinicMonthlyInvoiceUseCase,
)
from application.invoices.period_invoice_preview import build_period_invoice_preview
from models import Booking, InvoiceLine
from models.enums import BookingCreatedVia, BookingStatus
from tests.application.helpers.cancel_billable_c1_world import (
    PERIOD_MONTH,
    PERIOD_YEAR,
    RIDE_HT,
    _aug,
    add_canceled_booking,
    build_c1_world,
)
from tests.e2e.helpers.institution_invoice_plan_lha import HUG, LHA

FEE_PARTIAL = Decimal("45.00")
FEE_FULL = Decimal("90.00")
FEE_ZERO = Decimal("0.00")
MONEY = Decimal("0.01")


def build_c2_world(db) -> dict[str, Any]:
    return build_c1_world(db)


def add_completed_booking(db, world: dict[str, Any]) -> Booking:
    booking = Booking()
    booking.company_id = world["transport"].id
    booking.client_id = world["clinic_client"].id
    booking.customer_name = "Patient C2"
    booking.pickup_location = LHA
    booking.dropoff_location = HUG
    booking.scheduled_time = _aug(12)
    booking.completed_at = _aug(12, 11)
    booking.status = BookingStatus.COMPLETED.value
    booking.amount = RIDE_HT
    booking.billed_to_type = "clinic"
    booking.billing_party_id = world["clinic_bp"].id
    booking.billed_to_company_id = world["clinic"].id
    booking.created_via = BookingCreatedVia.INSTITUTION_PORTAL
    booking.is_return = False
    booking.is_cancellation_billable = None
    booking.cancellation_fee_amount = None
    booking.invoice_line_id = None
    db.session.add(booking)
    db.session.flush()
    return booking


def money(value: Any) -> Decimal:
    return Decimal(str(value)).quantize(MONEY)


def canonical_billable(booking: Booking) -> BillableAmount:
    return calculate_billable_booking_amount(booking)


def canonical_amount(booking: Booking) -> tuple[Decimal, str]:
    result = canonical_billable(booking)
    return money(result.amount_ht), result.source


def preview_clinic_amount(world: dict[str, Any], booking_id: int) -> Decimal | None:
    preview = build_period_invoice_preview(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        include_line_details=True,
    )
    for line in preview.preview_lines:
        if int(line.booking_id) == int(booking_id):
            return money(line.amount_ht)
    return None


def registry_clinic_amount(world: dict[str, Any]) -> Decimal | None:
    res = list_billing_opportunities(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
    )
    for item in res.clinic_items:
        if item.clinic_company_id == world["clinic"].id and item.transports_count > 0:
            return money(item.estimated_total)
    return None


def generate_clinic_amount(world: dict[str, Any], booking_id: int) -> Decimal | None:
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
            return money(line.line_total)
    return None


__all__ = [
    "FEE_FULL",
    "FEE_PARTIAL",
    "FEE_ZERO",
    "RIDE_HT",
    "add_canceled_booking",
    "add_completed_booking",
    "build_c2_world",
    "canonical_amount",
    "canonical_billable",
    "generate_clinic_amount",
    "money",
    "preview_clinic_amount",
    "registry_clinic_amount",
]
