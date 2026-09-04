"""Gate BE1–BE15 — batch patients idempotent (source = institution_invoice_plan)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from application.invoices.generate_invoice import (
    GenerateInvoiceInput,
    GenerateInvoiceUseCase,
)
from application.invoices.institution_invoice_eligibility import (
    reopen_market_lirie_validation_after_financial_change,
)
from application.invoices.institution_invoice_plan import build_institution_invoice_plan
from application.invoices.institution_patient_batch import (
    InstitutionPatientBatchInput,
    InstitutionPatientBatchUseCase,
)
from models import Invoice
from models.enums import InstitutionBillingControlStatus, InvoiceBillingStrategy
from tests.e2e.helpers.institution_invoice_plan_lha import (
    PERIOD_MONTH,
    PERIOD_YEAR,
    build_lha_august_2026_world,
)
from tests.e2e.helpers.institution_patient_batch_world import (
    extend_lha_world_for_patient_batch,
)

ZURICH = ZoneInfo("Europe/Zurich")
AUG_31 = datetime(2026, 8, 31, 23, 59, 59, tzinfo=ZURICH)
SEP_1 = datetime(2026, 9, 1, 0, 0, 0, tzinfo=ZURICH)


@pytest.fixture
def batch_world(db):
    return extend_lha_world_for_patient_batch(db, build_lha_august_2026_world(db))


def _pdf() -> MagicMock:
    pdf = MagicMock()
    pdf.generate_invoice_pdf.return_value = "https://cdn.example/invoice.pdf"
    return pdf


def _plan(world, *, now):
    return build_institution_invoice_plan(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        clinic_client_id=world["clinic_client"].id,
        now=now,
    )


def _input(world, patient_ids=None):
    return InstitutionPatientBatchInput(
        company_id=world["transport"].id,
        clinic_company_id=world["clinic"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_client_id=world["clinic_client"].id,
        institution_patient_ids=patient_ids,
    )


def _run(world, *, now, patient_ids=None):
    return InstitutionPatientBatchUseCase(pdf_service=_pdf()).execute(
        _input(world, patient_ids=patient_ids),
        now=now,
    )


def _s1_count(world) -> int:
    return Invoice.query.filter(
        Invoice.company_id == world["transport"].id,
        Invoice.period_year == PERIOD_YEAR,
        Invoice.period_month == PERIOD_MONTH,
        Invoice.billing_strategy == InvoiceBillingStrategy.S1_PATIENT,
    ).count()


def _bucket(plan, ipid: int):
    return next(p for p in plan.patients if p.institution_patient_id == ipid)


def _item(result, ipid: int):
    return next(i for i in result.invoices if i.patient_id == ipid)


class TestInstitutionPatientBatch:
    def test_plan_buckets_expose_booking_ids(self, db, batch_world):
        plan = _plan(batch_world, now=AUG_31)
        cavadini = batch_world["patients"]["cavadini"]
        bucket = _bucket(plan, cavadini.id)
        assert bucket.booking_ids == [
            batch_world["bookings"]["market_validated_patient"].id
        ]

    def test_be1_two_patients_two_invoices(self, db, batch_world):
        cavadini = batch_world["patients"]["cavadini"]
        dupont = batch_world["patients"]["dupont"]
        result = _run(batch_world, now=AUG_31, patient_ids=[cavadini.id, dupont.id])
        assert result.created_count == 2
        assert result.reused_count == 0
        assert result.failed_count == 0
        ids = {item.invoice_id for item in result.invoices}
        assert len(ids) == 2

    def test_be2_one_patient_multiple_bookings(self, db, batch_world):
        moretti = batch_world["patients"]["moretti"]
        result = _run(batch_world, now=AUG_31, patient_ids=[moretti.id])
        assert result.created_count == 1
        item = _item(result, moretti.id)
        assert set(item.booking_ids) == {
            batch_world["bookings"]["moretti_a"].id,
            batch_world["bookings"]["moretti_b"].id,
        }

    def test_be3_split_payer_patient_invoice_is_return_only(self, db, batch_world):
        dupont = batch_world["patients"]["dupont"]
        result = _run(batch_world, now=AUG_31, patient_ids=[dupont.id])
        item = _item(result, dupont.id)
        assert item.booking_ids == [batch_world["bookings"]["ar_split_ret"].id]
        assert batch_world["bookings"]["ar_split_out"].id not in item.booking_ids

    def test_be4_round_trip_same_patient_payer_kept(self, db, batch_world):
        rivet = batch_world["patients"]["rivet"]
        result = _run(batch_world, now=AUG_31, patient_ids=[rivet.id])
        assert result.created_count == 1
        item = _item(result, rivet.id)
        assert set(item.booking_ids) == {
            batch_world["bookings"]["rivet_out"].id,
            batch_world["bookings"]["rivet_ret"].id,
        }

    def test_be5_sequential_identical_call_reuses(self, db, batch_world):
        cavadini = batch_world["patients"]["cavadini"]
        dupont = batch_world["patients"]["dupont"]
        ids = [cavadini.id, dupont.id]
        first = _run(batch_world, now=AUG_31, patient_ids=ids)
        assert first.created_count == 2
        assert first.reused_count == 0
        before = _s1_count(batch_world)
        second = _run(batch_world, now=AUG_31, patient_ids=ids)
        assert second.created_count == 0
        assert second.reused_count == 2
        assert _s1_count(batch_world) == before
        assert {i.invoice_id for i in first.invoices} == {
            i.invoice_id for i in second.invoices
        }

    def test_be6_concurrent_no_duplicate(self, app, db, batch_world):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        cavadini = batch_world["patients"]["cavadini"]
        moretti = batch_world["patients"]["moretti"]
        inp = _input(batch_world, patient_ids=[cavadini.id, moretti.id])
        barrier = Barrier(2)

        def _once():
            barrier.wait(timeout=30)
            with app.app_context():
                return InstitutionPatientBatchUseCase(pdf_service=_pdf()).execute(
                    inp, now=AUG_31
                )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = [
                future.result() for future in [pool.submit(_once), pool.submit(_once)]
            ]

        totals = [(r.created_count, r.reused_count, r.failed_count) for r in results]
        assert all(failed == 0 for _, _, failed in totals), totals
        assert sum(c + r for c, r, _ in totals) == 4
        assert _s1_count(batch_world) == 2
        invoice_ids = {
            item.invoice_id for result in results for item in result.invoices
        }
        assert len(invoice_ids) == 2

    def test_be7_retry_after_lost_response_reuses(self, db, batch_world):
        moretti = batch_world["patients"]["moretti"]
        first = _run(batch_world, now=AUG_31, patient_ids=[moretti.id])
        assert first.created_count == 1
        lost = first
        retry = _run(batch_world, now=AUG_31, patient_ids=[moretti.id])
        assert retry.created_count == 0
        assert retry.reused_count == 1
        assert retry.invoices[0].invoice_id == lost.invoices[0].invoice_id

    def test_be8_already_invoiced_booking_not_duplicated(self, db, batch_world):
        cavadini = batch_world["patients"]["cavadini"]
        booking_id = batch_world["bookings"]["market_validated_patient"].id
        prior = GenerateInvoiceUseCase(pdf_service=_pdf()).execute(
            GenerateInvoiceInput(
                company_id=batch_world["transport"].id,
                client_id=batch_world["clinic_client"].id,
                period_year=PERIOD_YEAR,
                period_month=PERIOD_MONTH,
                billing_party_id=batch_world["patient_bps"]["cavadini"].id,
                reservation_ids=[booking_id],
                institution_patient_id=cavadini.id,
                strict_reservation_ids=True,
            ),
            now=AUG_31,
        )
        assert prior.success is True
        result = _run(
            batch_world,
            now=AUG_31,
            patient_ids=[cavadini.id, batch_world["patients"]["dupont"].id],
        )
        cavadini_item = _item(result, cavadini.id)
        assert cavadini_item.result == "existing"
        assert cavadini_item.invoice_id == prior.invoice_id
        assert cavadini_item.booking_ids.count(booking_id) == 1
        assert result.invoices

    def test_be9_unchecked_patient_not_invoiced(self, db, batch_world):
        cavadini = batch_world["patients"]["cavadini"]
        moretti = batch_world["patients"]["moretti"]
        result = _run(batch_world, now=AUG_31, patient_ids=[cavadini.id])
        assert all(item.patient_id != moretti.id for item in result.invoices)
        assert result.skipped_count >= 1

    def test_be10_pending_market_excluded(self, db, batch_world):
        rossi = batch_world["patients"]["rossi"]
        plan = _plan(batch_world, now=AUG_31)
        assert all(p.institution_patient_id != rossi.id for p in plan.patients)
        result = _run(batch_world, now=AUG_31, patient_ids=[rossi.id])
        assert result.created_count == 0
        assert all(item.patient_id != rossi.id for item in result.invoices)

    def test_be11_disputed_market_excluded(self, db, batch_world):
        bianchi = batch_world["patients"]["bianchi"]
        plan = _plan(batch_world, now=AUG_31)
        assert all(p.institution_patient_id != bianchi.id for p in plan.patients)
        result = _run(batch_world, now=AUG_31, patient_ids=[bianchi.id])
        assert result.created_count == 0

    def test_be12_auto_released_patient_included(self, db, batch_world):
        verdi = batch_world["patients"]["verdi"]
        blocked = _plan(batch_world, now=AUG_31)
        assert all(p.institution_patient_id != verdi.id for p in blocked.patients)
        result = _run(batch_world, now=SEP_1, patient_ids=[verdi.id])
        assert result.created_count == 1
        item = _item(result, verdi.id)
        assert item.booking_ids == [batch_world["bookings"]["verdi_pending"].id]

    def test_be13_financial_reopen_excludes_until_eligible(self, db, batch_world):
        faure = batch_world["patients"]["faure"]
        booking = batch_world["bookings"]["faure_reopen"]
        booking.amount = Decimal("80.00")
        assert reopen_market_lirie_validation_after_financial_change(booking) is True
        db.session.flush()
        assert (
            booking.institution_control_status
            == InstitutionBillingControlStatus.PENDING_REVIEW
        )
        plan = _plan(batch_world, now=AUG_31)
        assert all(p.institution_patient_id != faure.id for p in plan.patients)
        result = _run(batch_world, now=AUG_31, patient_ids=[faure.id])
        assert result.created_count == 0

    def test_be14_be15_conservation_ids_and_totals(self, db, batch_world):
        plan = _plan(batch_world, now=AUG_31)
        selected = [
            batch_world["patients"]["cavadini"].id,
            batch_world["patients"]["dupont"].id,
            batch_world["patients"]["moretti"].id,
            batch_world["patients"]["rivet"].id,
        ]
        buckets = [
            p for p in plan.patients if p.institution_patient_id in set(selected)
        ]
        expected_ids = {bid for bucket in buckets for bid in bucket.booking_ids}
        expected_ht = round(sum(float(bucket.estimated_total) for bucket in buckets), 2)
        result = _run(batch_world, now=AUG_31, patient_ids=selected)
        assert result.created_count == len(selected)
        assert result.failed_count == 0
        got_ids = {bid for item in result.invoices for bid in item.booking_ids}
        got_ht = round(sum(item.total_ht for item in result.invoices), 2)
        assert got_ids == expected_ids
        assert got_ht == expected_ht

    def test_second_call_db_invoice_count_unchanged(self, db, batch_world):
        selected = [
            batch_world["patients"]["cavadini"].id,
            batch_world["patients"]["moretti"].id,
        ]
        first = _run(batch_world, now=AUG_31, patient_ids=selected)
        assert first.created_count == 2
        assert first.reused_count == 0
        count_after_first = _s1_count(batch_world)
        second = _run(batch_world, now=AUG_31, patient_ids=selected)
        assert second.created_count == 0
        assert second.reused_count == 2
        assert _s1_count(batch_world) == count_after_first
