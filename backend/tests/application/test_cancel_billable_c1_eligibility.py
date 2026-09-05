"""C1 — éligibilité annulation facturable. Tests first, pas de C2/C3/C4."""

from __future__ import annotations

from unittest.mock import MagicMock

from application.invoices.billing_opportunities import list_billing_opportunities
from application.invoices.generate_clinic_monthly_invoice import (
    GenerateClinicMonthlyInvoiceInput,
    GenerateClinicMonthlyInvoiceUseCase,
)
from application.invoices.generate_invoice import (
    GenerateInvoiceInput,
    GenerateInvoiceUseCase,
)
from application.invoices.period_invoice_preview import build_period_invoice_preview
from ext import db
from models import InvoiceLine
from tests.application.helpers.cancel_billable_c1_world import (
    PERIOD_MONTH,
    PERIOD_YEAR,
    add_canceled_booking,
    add_client_stay,
    build_c1_world,
)


def _preview_clinic_ids(world) -> set[int]:
    preview = build_period_invoice_preview(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        include_line_details=True,
    )
    return {int(line.booking_id) for line in preview.preview_lines}


def _preview_patient_ids(world) -> set[int]:
    preview = build_period_invoice_preview(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        client_id=world["clinic_client"].id,
        include_line_details=True,
    )
    return {int(line.booking_id) for line in preview.preview_lines}


def _generate_clinic_ids(world) -> set[int]:
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
        return set()
    ids: set[int] = set()
    for line in InvoiceLine.query.filter_by(invoice_id=result.invoice_id).all():
        meta = line.line_meta if isinstance(line.line_meta, dict) else {}
        claimed = meta.get("booking_ids") or []
        if claimed:
            ids.update(int(i) for i in claimed)
        elif line.reservation_id is not None:
            ids.add(int(line.reservation_id))
    return ids


def _generate_patient_ids(world, reservation_ids: list[int]) -> set[int]:
    pdf = MagicMock()
    pdf.generate_invoice_pdf.return_value = "https://cdn.example/invoice.pdf"
    result = GenerateInvoiceUseCase(pdf_service=pdf).execute(
        GenerateInvoiceInput(
            company_id=world["transport"].id,
            client_id=world["clinic_client"].id,
            period_year=PERIOD_YEAR,
            period_month=PERIOD_MONTH,
            billing_party_id=world["clinic_bp"].id,
            reservation_ids=reservation_ids,
        )
    )
    if not result.success or result.invoice_id is None:
        return set()
    ids: set[int] = set()
    for line in InvoiceLine.query.filter_by(invoice_id=result.invoice_id).all():
        if line.reservation_id is not None:
            ids.add(int(line.reservation_id))
    return ids


def _clinic_register_has_transports(world) -> bool:
    res = list_billing_opportunities(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
    )
    return any(
        item.clinic_company_id == world["clinic"].id and item.transports_count > 0
        for item in res.clinic_items
    )


def _assert_clinic_surfaces(world, booking, *, included: bool) -> None:
    """Registre et preview avant generate : l'émission pose invoice_line_id."""
    db.session.flush()
    bid = int(booking.id)
    register_on = _clinic_register_has_transports(world)
    preview_ids = _preview_clinic_ids(world)
    generate_ids = _generate_clinic_ids(world)
    assert preview_ids == generate_ids
    if included:
        assert register_on is True
        assert bid in preview_ids
        assert bid in generate_ids
    else:
        assert bid not in preview_ids
        assert bid not in generate_ids


def test_c1_canceled_not_billable_excluded(db):
    world = build_c1_world(db)
    booking = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=False,
    )
    add_client_stay(
        db,
        client_id=world["clinic_client"].id,
        clinic_id=world["clinic"].id,
        when=booking.scheduled_time,
    )
    _assert_clinic_surfaces(world, booking, included=False)


def test_c1_canceled_billable_patient_included(db):
    world = build_c1_world(db)
    booking = add_canceled_booking(
        db,
        world,
        billed_to_type="patient",
        is_cancellation_billable=True,
    )
    db.session.flush()
    assert int(booking.id) in _preview_patient_ids(world)
    assert int(booking.id) in _generate_patient_ids(world, [int(booking.id)])


def test_c1_canceled_billable_clinic_with_stay_included(db):
    world = build_c1_world(db)
    booking = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
    )
    add_client_stay(
        db,
        client_id=world["clinic_client"].id,
        clinic_id=world["clinic"].id,
        when=booking.scheduled_time,
    )
    _assert_clinic_surfaces(world, booking, included=True)


def test_c1_canceled_billable_clinic_without_stay_included(db):
    """P0 : rattachement clinique explicite prime sur l'absence de ClientStay."""
    world = build_c1_world(db)
    booking = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
    )
    assert booking.billed_to_company_id == world["clinic"].id
    _assert_clinic_surfaces(world, booking, included=True)


def test_c1_override_preview_and_generate_agree(db):
    world = build_c1_world(db)
    booking = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=False,
        billing_override_reason="Décision commerciale — frais dus",
    )
    add_client_stay(
        db,
        client_id=world["clinic_client"].id,
        clinic_id=world["clinic"].id,
        when=booking.scheduled_time,
    )
    db.session.flush()
    bid = int(booking.id)
    preview_ids = _preview_clinic_ids(world)
    generate_ids = _generate_clinic_ids(world)
    assert preview_ids == generate_ids
    assert bid in preview_ids
    assert bid in generate_ids


def test_c1_round_trip_fee_on_outbound_only(db):
    world = build_c1_world(db)
    outbound = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
        is_return=False,
        day=12,
    )
    inbound = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
        is_return=True,
        parent_booking_id=int(outbound.id),
        day=12,
    )
    add_client_stay(
        db,
        client_id=world["clinic_client"].id,
        clinic_id=world["clinic"].id,
        when=outbound.scheduled_time,
    )
    db.session.flush()
    preview_ids = _preview_clinic_ids(world)
    generate_ids = _generate_clinic_ids(world)
    assert preview_ids == generate_ids
    assert int(outbound.id) in preview_ids
    assert int(outbound.id) in generate_ids
    assert int(inbound.id) not in preview_ids
    assert int(inbound.id) not in generate_ids
