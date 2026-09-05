"""C2b — consommateur unresolved. Pas de C1/C3/C4."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock

from application.invoices.billable_amount import (
    SOURCE_CANCELLATION_FEE,
    UNRESOLVED_CANCELLATION_REASON,
    calculate_billable_booking_amount,
)
from application.invoices.billing_opportunities import list_billing_opportunities
from application.invoices.generate_clinic_monthly_invoice import (
    GenerateClinicMonthlyInvoiceInput,
    GenerateClinicMonthlyInvoiceUseCase,
)
from application.invoices.institution_invoice_plan import build_institution_invoice_plan
from application.invoices.period_invoice_preview import build_period_invoice_preview
from models import Invoice, InvoiceLine
from tests.application.helpers.cancel_billable_c1_world import (
    PERIOD_MONTH,
    PERIOD_YEAR,
    add_canceled_booking,
    build_c1_world,
)
from tests.application.helpers.cancel_billable_c2_world import add_completed_booking

COMPLETED_HT = Decimal("320.00")
CANCELED_RIDE_HT = Decimal("40.00")
RESOLVED_FEE_HT = Decimal("35.00")
EXPECTED_WITHOUT_FEE = COMPLETED_HT
EXPECTED_WITH_FEE = COMPLETED_HT + RESOLVED_FEE_HT


def _world_with_pair(db, *, fee):
    world = build_c1_world(db)
    completed = add_completed_booking(db, world)
    completed.amount = COMPLETED_HT
    canceled = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
        cancellation_fee_amount=fee,
        day=13,
    )
    canceled.amount = CANCELED_RIDE_HT
    db.session.flush()
    world["completed"] = completed
    world["canceled"] = canceled
    return world


def _preview(world):
    return build_period_invoice_preview(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        include_line_details=True,
    )


def _registry_total(world) -> Decimal | None:
    res = list_billing_opportunities(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
    )
    for item in res.clinic_items:
        if item.clinic_company_id == world["clinic"].id:
            return Decimal(str(item.estimated_total)).quantize(Decimal("0.01"))
    return None


def _generate(world):
    pdf = MagicMock()
    pdf.generate_invoice_pdf.return_value = "https://cdn.example/invoice.pdf"
    return GenerateClinicMonthlyInvoiceUseCase(pdf_service=pdf).execute(
        GenerateClinicMonthlyInvoiceInput(
            company_id=world["transport"].id,
            clinic_company_id=world["clinic"].id,
            period_year=PERIOD_YEAR,
            period_month=PERIOD_MONTH,
        )
    )


def test_c2b_unresolved_excluded_other_rides_still_invoiced(db):
    world = _world_with_pair(db, fee=None)
    completed_id = int(world["completed"].id)
    canceled_id = int(world["canceled"].id)

    billed = calculate_billable_booking_amount(world["canceled"])
    assert billed.resolved is False
    assert billed.source != "booking.amount"

    preview = _preview(world)
    preview_ids = {int(line.booking_id) for line in preview.preview_lines}
    unresolved = (preview.eligibility or {}).get("cancellation_fee_unresolved") or {}
    assert Decimal(str(preview.estimated_total)) == EXPECTED_WITHOUT_FEE
    assert completed_id in preview_ids
    assert canceled_id not in preview_ids
    assert not any(float(line.amount_ht) == 0 for line in preview.preview_lines)
    assert canceled_id in set(unresolved.get("booking_ids") or [])
    assert UNRESOLVED_CANCELLATION_REASON in " ".join(preview.warnings).lower()

    assert _registry_total(world) == EXPECTED_WITHOUT_FEE
    plan = build_institution_invoice_plan(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        clinic_client_id=world["clinic_client"].id,
    )
    assert plan.clinic is not None
    assert Decimal(str(plan.clinic.estimated_total)) == EXPECTED_WITHOUT_FEE

    result = _generate(world)
    assert result.success is True
    assert result.invoice_id is not None
    invoice = db.session.get(Invoice, result.invoice_id)
    assert Decimal(str(invoice.total_amount)) == EXPECTED_WITHOUT_FEE
    line_booking_ids: set[int] = set()
    for line in InvoiceLine.query.filter_by(invoice_id=result.invoice_id).all():
        assert Decimal(str(line.line_total)) != Decimal("0.00")
        meta = line.line_meta if isinstance(line.line_meta, dict) else {}
        line_booking_ids.update(int(i) for i in (meta.get("booking_ids") or []))
        if line.reservation_id is not None:
            line_booking_ids.add(int(line.reservation_id))
    assert completed_id in line_booking_ids
    assert canceled_id not in line_booking_ids


def test_c2b_resolved_fee_enters_invoice_total(db):
    world = _world_with_pair(db, fee=RESOLVED_FEE_HT)
    canceled_id = int(world["canceled"].id)
    billed = calculate_billable_booking_amount(world["canceled"])
    assert billed.resolved is True
    assert billed.source == SOURCE_CANCELLATION_FEE
    assert billed.amount_ht == RESOLVED_FEE_HT

    preview = _preview(world)
    preview_ids = {int(line.booking_id) for line in preview.preview_lines}
    assert Decimal(str(preview.estimated_total)) == EXPECTED_WITH_FEE
    assert canceled_id in preview_ids
    assert _registry_total(world) == EXPECTED_WITH_FEE

    result = _generate(world)
    assert result.success is True
    invoice = db.session.get(Invoice, result.invoice_id)
    assert Decimal(str(invoice.total_amount)) == EXPECTED_WITH_FEE
    claimed: set[int] = set()
    for line in InvoiceLine.query.filter_by(invoice_id=result.invoice_id).all():
        meta = line.line_meta if isinstance(line.line_meta, dict) else {}
        claimed.update(int(i) for i in (meta.get("booking_ids") or []))
        if line.reservation_id is not None:
            claimed.add(int(line.reservation_id))
    assert canceled_id in claimed
