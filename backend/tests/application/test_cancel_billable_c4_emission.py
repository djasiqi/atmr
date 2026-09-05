"""C4 — émission PDF / QR. Chaîne réelle, pas de mock PDF.

state → eligibility → amount → description → preview → InvoiceLine → PDF → QR
"""

from __future__ import annotations

from decimal import Decimal

from application.invoices.billable_amount import (
    SOURCE_CANCELLATION_FEE,
    SOURCE_CANCELLATION_UNRESOLVED,
    calculate_billable_booking_amount,
)
from tests.application.helpers.cancel_billable_c1_world import add_canceled_booking
from tests.application.helpers.cancel_billable_c2_world import add_completed_booking
from tests.application.helpers.cancel_billable_c4_world import (
    TRAJET_EFFECTUE_TOKEN,
    add_canceled_emission_booking,
    billable_source,
    build_c4_world,
    expected_cancellation_label,
    generate_real_invoice,
    invoice_line_ids_and_total,
    money,
    plan,
    preview,
    qr_amount,
    read_pdf_text,
)

FEE_PARTIAL = Decimal("45.00")
FEE_FULL = Decimal("90.00")
RIDE_HT = Decimal("90.00")
COMPLETED_HT = Decimal("320.00")
NO_SHOW_LABEL = "Client ne s'est pas présenté"


def test_c4_partial_no_show_pdf_qr_uses_fee_not_ride(db):
    world = build_c4_world(db)
    booking = add_canceled_emission_booking(
        db, world, fee_amount=FEE_PARTIAL, reason_code="NO_SHOW"
    )
    billed = calculate_billable_booking_amount(booking)
    assert billed.resolved is True
    assert billed.amount_ht == FEE_PARTIAL
    assert billed.source == SOURCE_CANCELLATION_FEE
    assert billed.source != "booking.amount"

    motif = expected_cancellation_label(booking)
    assert motif == NO_SHOW_LABEL

    prev = preview(world)
    assert money(prev.estimated_total) == FEE_PARTIAL
    preview_line = next(
        line for line in prev.preview_lines if int(line.booking_id) == int(booking.id)
    )
    assert str(preview_line.description) == motif
    assert TRAJET_EFFECTUE_TOKEN not in str(preview_line.description)

    clinic_plan = plan(world)
    assert clinic_plan.clinic is not None
    assert money(clinic_plan.clinic.estimated_total) == FEE_PARTIAL

    invoice = generate_real_invoice(world)
    issued_ids, issued_ht, descriptions = invoice_line_ids_and_total(int(invoice.id))
    assert int(booking.id) in issued_ids
    assert issued_ht == FEE_PARTIAL
    assert money(invoice.total_amount) == FEE_PARTIAL
    assert descriptions == [motif]
    assert all(TRAJET_EFFECTUE_TOKEN not in desc for desc in descriptions)

    pdf_text = read_pdf_text(invoice)
    assert NO_SHOW_LABEL in pdf_text
    assert TRAJET_EFFECTUE_TOKEN not in pdf_text
    assert "45.00" in pdf_text or "45,00" in pdf_text

    assert qr_amount(invoice) == FEE_PARTIAL
    assert billable_source(booking) == SOURCE_CANCELLATION_FEE


def test_c4_unresolved_fee_never_emits_zero_line(db):
    world = build_c4_world(db)
    completed = add_completed_booking(db, world)
    completed.amount = COMPLETED_HT
    canceled = add_canceled_booking(
        db,
        world,
        billed_to_type="clinic",
        is_cancellation_billable=True,
        cancellation_fee_amount=None,
        day=13,
    )
    canceled.amount = Decimal("40.00")
    db.session.flush()

    billed = calculate_billable_booking_amount(canceled)
    assert billed.resolved is False
    assert billed.source == SOURCE_CANCELLATION_UNRESOLVED
    assert billed.source != "booking.amount"

    prev = preview(world)
    preview_ids = {int(line.booking_id) for line in prev.preview_lines}
    assert money(prev.estimated_total) == COMPLETED_HT
    assert int(completed.id) in preview_ids
    assert int(canceled.id) not in preview_ids
    assert not any(
        money(line.amount_ht) == Decimal("0.00") for line in prev.preview_lines
    )

    clinic_plan = plan(world)
    assert clinic_plan.clinic is not None
    assert money(clinic_plan.clinic.estimated_total) == COMPLETED_HT

    invoice = generate_real_invoice(world)
    issued_ids, issued_ht, descriptions = invoice_line_ids_and_total(int(invoice.id))
    assert int(completed.id) in issued_ids
    assert int(canceled.id) not in issued_ids
    assert issued_ht == COMPLETED_HT
    assert money(invoice.total_amount) == COMPLETED_HT
    assert issued_ht != Decimal("0.00")
    assert all(desc.strip() != "0.00" for desc in descriptions)

    pdf_text = read_pdf_text(invoice)
    assert NO_SHOW_LABEL not in pdf_text
    assert qr_amount(invoice) == COMPLETED_HT


def test_c4_full_fare_explicit_fee_not_booking_amount(db):
    world = build_c4_world(db)
    booking = add_canceled_emission_booking(
        db, world, fee_amount=FEE_FULL, reason_code="NO_SHOW"
    )
    billed = calculate_billable_booking_amount(booking)
    assert billed.amount_ht == FEE_FULL
    assert billed.source == SOURCE_CANCELLATION_FEE
    assert Decimal(str(booking.amount)) == RIDE_HT
    assert billed.source != "booking.amount"

    motif = expected_cancellation_label(booking)
    prev = preview(world)
    assert money(prev.estimated_total) == FEE_FULL
    preview_line = next(
        line for line in prev.preview_lines if int(line.booking_id) == int(booking.id)
    )
    assert str(preview_line.description) == motif

    invoice = generate_real_invoice(world)
    issued_ids, issued_ht, descriptions = invoice_line_ids_and_total(int(invoice.id))
    assert int(booking.id) in issued_ids
    assert issued_ht == FEE_FULL
    assert money(invoice.total_amount) == FEE_FULL
    assert descriptions == [motif]
    assert all(TRAJET_EFFECTUE_TOKEN not in desc for desc in descriptions)

    pdf_text = read_pdf_text(invoice)
    assert NO_SHOW_LABEL in pdf_text
    assert TRAJET_EFFECTUE_TOKEN not in pdf_text
    assert qr_amount(invoice) == FEE_FULL
    assert billable_source(booking) == SOURCE_CANCELLATION_FEE
