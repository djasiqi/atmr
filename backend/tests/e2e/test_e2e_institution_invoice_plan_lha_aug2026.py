"""Gate E2E — institution_invoice_plan LHA / août 2026.

Chaîne : booking → eligibility → plan → preview → draft.
Horloge déterministe Europe/Zurich. Pas de batch patients.

Exécution :
    docker compose -f docker-compose.test.yml run --rm backend_tests sh -c \\
      "flask db upgrade heads && python -m pytest \\
       tests/e2e/test_e2e_institution_invoice_plan_lha_aug2026.py -v"
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from application.invoices.generate_clinic_monthly_invoice import (
    GenerateClinicMonthlyInvoiceInput,
    GenerateClinicMonthlyInvoiceUseCase,
)
from application.invoices.generate_invoice import (
    GenerateInvoiceInput,
    GenerateInvoiceUseCase,
)
from application.invoices.institution_invoice_eligibility import (
    market_lirie_deadline,
    reopen_market_lirie_validation_after_financial_change,
)
from application.invoices.institution_invoice_plan import build_institution_invoice_plan
from application.invoices.period_invoice_preview import build_period_invoice_preview
from models import InvoiceLine
from models.enums import InstitutionBillingControlStatus
from tests.e2e.helpers.institution_invoice_plan_lha import (
    PERIOD_MONTH,
    PERIOD_YEAR,
    build_lha_august_2026_world,
)

ZURICH = ZoneInfo("Europe/Zurich")
AUG_31 = datetime(2026, 8, 31, 23, 59, 59, tzinfo=ZURICH)
SEP_1 = datetime(2026, 9, 1, 0, 0, 0, tzinfo=ZURICH)
AUG_20 = datetime(2026, 8, 20, 12, 0, tzinfo=ZURICH)

pytestmark = pytest.mark.e2e


@pytest.fixture
def lha_world(db):
    return build_lha_august_2026_world(db)


def _plan(world, *, now):
    return build_institution_invoice_plan(
        company_id=world["transport"].id,
        period_year=PERIOD_YEAR,
        period_month=PERIOD_MONTH,
        clinic_company_id=world["clinic"].id,
        clinic_client_id=world["clinic_client"].id,
        now=now,
    )


def _ids(world, *labels: str) -> set[int]:
    return {int(world["bookings"][label].id) for label in labels}


def _ledger(plan) -> dict:
    rec = plan.reconciliation or {}
    assert rec, "le plan doit exposer une réconciliation"
    return rec


def _audits_by_label(world, ledger: dict) -> dict[str, dict]:
    by_id = {int(row["booking_id"]): row for row in ledger.get("bookings") or []}
    out = {}
    for label, booking in world["bookings"].items():
        row = by_id.get(int(booking.id))
        assert row is not None, f"{label} (#{booking.id}) absent du registre"
        out[label] = row
    return out


def _bucket_ids(ledger: dict, name: str) -> set[int]:
    return {int(i) for i in (ledger["buckets"][name]["booking_ids"] or [])}


def _assert_conservation(ledger: dict, expected_ids: set[int]) -> None:
    assert ledger["conservation_ok"] is True
    assert not ledger["duplicate_booking_ids"]
    assert not ledger["missing_from_buckets"]
    seen: list[int] = []
    for bucket in ledger["buckets"].values():
        seen.extend(int(i) for i in bucket["booking_ids"])
    assert set(seen) == expected_ids
    assert len(seen) == len(expected_ids)
    bucket_sum = round(
        sum(float(b["amount_ht"]) for b in ledger["buckets"].values()), 2
    )
    assert bucket_sum == float(ledger["considered_amount_ht"])


class TestE2EInstitutionInvoicePlanLhaAug2026:
    def test_reconciliation_and_clock(self, db, lha_world):
        all_ids = {int(b.id) for b in lha_world["bookings"].values()}
        clinic_open = _ids(
            lha_world,
            "portfolio_clinic",
            "market_validated_clinic",
            "ar_same_out",
            "ar_same_ret",
            "ar_split_out",
            "same_day_a",
            "same_day_b",
            "financial_reopen",
        )
        patient_open = _ids(lha_world, "market_validated_patient", "ar_split_ret")
        pending = _ids(lha_world, "market_pending")
        disputed = _ids(lha_world, "market_disputed")

        plan_aug = _plan(lha_world, now=AUG_31)
        rec_aug = _ledger(plan_aug)
        _assert_conservation(rec_aug, all_ids)
        assert _bucket_ids(rec_aug, "clinic_billable") == clinic_open
        assert _bucket_ids(rec_aug, "patient_billable") == patient_open
        assert _bucket_ids(rec_aug, "pending_blocked") == pending
        assert _bucket_ids(rec_aug, "disputed_blocked") == disputed
        assert rec_aug["buckets"]["clinic_billable"]["amount_ht"] == 320.0
        assert rec_aug["buckets"]["patient_billable"]["amount_ht"] == 80.0
        assert rec_aug["buckets"]["pending_blocked"]["amount_ht"] == 40.0
        assert rec_aug["buckets"]["disputed_blocked"]["amount_ht"] == 40.0
        assert rec_aug["considered_amount_ht"] == 480.0

        audits = _audits_by_label(lha_world, rec_aug)
        assert audits["portfolio_clinic"]["origin"] == "OWN_PORTFOLIO"
        assert audits["portfolio_clinic"]["validation_status"] == "not_required"
        assert audits["portfolio_clinic"]["payer"] == "clinic"
        assert audits["portfolio_clinic"]["eligible"] is True
        assert audits["market_pending"]["origin"] == "LIRIE_MARKETPLACE"
        assert audits["market_pending"]["validation_status"] == "pending"
        assert audits["market_pending"]["eligible"] is False
        assert audits["market_pending"]["exclusion_reason"] == (
            "market_pending_before_deadline"
        )
        assert audits["market_disputed"]["validation_status"] == "disputed"
        assert audits["market_disputed"]["eligible"] is False
        assert audits["ar_split_out"]["payer"] == "clinic"
        assert audits["ar_split_ret"]["payer"] == "patient"
        assert audits["ar_same_out"]["grouping_relation"] == "parent_booking_id" or (
            audits["ar_same_ret"]["grouping_relation"] == "parent_booking_id"
        )
        assert audits["same_day_a"]["grouping_relation"] in (None, "request_id")
        assert audits["same_day_b"]["grouping_relation"] in (None, "request_id")
        assert audits["same_day_a"]["group_id"] != audits["same_day_b"]["group_id"] or (
            audits["same_day_a"]["grouping_relation"] is None
        )

        plan_sep = _plan(lha_world, now=SEP_1)
        rec_sep = _ledger(plan_sep)
        _assert_conservation(rec_sep, all_ids)
        assert _bucket_ids(rec_sep, "clinic_billable") == clinic_open | pending
        assert _bucket_ids(rec_sep, "pending_blocked") == set()
        assert _bucket_ids(rec_sep, "disputed_blocked") == disputed
        sep_pending = next(
            row
            for row in rec_sep["bookings"]
            if row["booking_id"] == lha_world["bookings"]["market_pending"].id
        )
        assert sep_pending["validation_status"] == "auto_released"
        assert sep_pending["persisted_control_status"] == "pending_review"
        assert sep_pending["eligible"] is True
        disputed_sep = next(
            row
            for row in rec_sep["bookings"]
            if row["booking_id"] == lha_world["bookings"]["market_disputed"].id
        )
        assert disputed_sep["validation_status"] == "disputed"
        assert disputed_sep["eligible"] is False

        feb = market_lirie_deadline(datetime(2026, 2, 10, 8, 0, tzinfo=ZURICH))
        assert feb.day == 28
        assert feb.month == 2

    def test_round_trip_grouping_in_preview(self, db, lha_world):
        preview = build_period_invoice_preview(
            company_id=lha_world["transport"].id,
            period_year=PERIOD_YEAR,
            period_month=PERIOD_MONTH,
            clinic_company_id=lha_world["clinic"].id,
            include_line_details=True,
            now=SEP_1,
        )
        same_out = lha_world["bookings"]["ar_same_out"].id
        same_ret = lha_world["bookings"]["ar_same_ret"].id
        split_out = lha_world["bookings"]["ar_split_out"].id
        split_ret = lha_world["bookings"]["ar_split_ret"].id
        day_a = lha_world["bookings"]["same_day_a"].id
        day_b = lha_world["bookings"]["same_day_b"].id

        by_primary = {int(pl.booking_id): pl for pl in preview.preview_lines}
        merged = next(
            pl
            for pl in preview.preview_lines
            if {int(pl.booking_id), int(pl.round_trip_partner_booking_id or -1)}
            == {same_out, same_ret}
        )
        assert merged.is_round_trip_leg is True
        assert {merged.booking_id, merged.round_trip_partner_booking_id} == {
            same_out,
            same_ret,
        }
        assert float(merged.amount_ht) == 80.0

        assert split_out in by_primary
        assert by_primary[split_out].is_round_trip_leg is False
        assert split_ret not in by_primary

        assert day_a in by_primary
        assert day_b in by_primary
        assert by_primary[day_a].is_round_trip_leg is False
        assert by_primary[day_b].is_round_trip_leg is False

    def test_financial_reopen_blocks_before_deadline(self, db, lha_world):
        booking = lha_world["bookings"]["financial_reopen"]
        booking.amount = Decimal("80.00")
        assert reopen_market_lirie_validation_after_financial_change(booking) is True
        db.session.flush()
        assert (
            booking.institution_control_status
            == InstitutionBillingControlStatus.PENDING_REVIEW
        )

        plan = _plan(lha_world, now=AUG_20)
        rec = _ledger(plan)
        assert int(booking.id) in _bucket_ids(rec, "pending_blocked")
        row = next(r for r in rec["bookings"] if r["booking_id"] == booking.id)
        assert row["validation_status"] == "pending"
        assert row["eligible"] is False
        assert row["persisted_control_status"] != "validated"

    def test_preview_and_draft_match_plan(self, db, lha_world):
        plan = _plan(lha_world, now=SEP_1)
        rec = _ledger(plan)
        clinic_ids = _bucket_ids(rec, "clinic_billable")
        clinic_ht = float(rec["buckets"]["clinic_billable"]["amount_ht"])
        assert clinic_ids
        assert float(plan.clinic.estimated_total) == clinic_ht
        assert plan.clinic.transports_count == len(clinic_ids)

        preview = build_period_invoice_preview(
            company_id=lha_world["transport"].id,
            period_year=PERIOD_YEAR,
            period_month=PERIOD_MONTH,
            clinic_company_id=lha_world["clinic"].id,
            include_line_details=True,
            now=SEP_1,
        )
        preview_ids: set[int] = set()
        for pl in preview.preview_lines:
            preview_ids.add(int(pl.booking_id))
            if pl.round_trip_partner_booking_id is not None:
                preview_ids.add(int(pl.round_trip_partner_booking_id))
        assert preview_ids == clinic_ids
        assert float(preview.estimated_total) == clinic_ht

        pdf = MagicMock()
        pdf.generate_invoice_pdf.return_value = "https://cdn.example/invoice.pdf"
        result = GenerateClinicMonthlyInvoiceUseCase(pdf_service=pdf).execute(
            GenerateClinicMonthlyInvoiceInput(
                company_id=lha_world["transport"].id,
                clinic_company_id=lha_world["clinic"].id,
                period_year=PERIOD_YEAR,
                period_month=PERIOD_MONTH,
            ),
            now=SEP_1,
        )
        assert result.success is True, result.error
        assert result.invoice_id is not None
        draft_ids: set[int] = set()
        draft_ht = 0.0
        for line in InvoiceLine.query.filter_by(invoice_id=result.invoice_id).all():
            draft_ht = round(draft_ht + float(line.line_total), 2)
            meta = line.line_meta if isinstance(line.line_meta, dict) else {}
            ids = meta.get("booking_ids") or []
            if ids:
                draft_ids.update(int(i) for i in ids)
            elif line.reservation_id is not None:
                draft_ids.add(int(line.reservation_id))
        assert draft_ids == clinic_ids
        assert draft_ht == clinic_ht

        patient_bucket = next(
            p
            for p in plan.patients
            if p.institution_patient_id == lha_world["patients"]["cavadini"].id
        )
        patient_preview = build_period_invoice_preview(
            company_id=lha_world["transport"].id,
            period_year=PERIOD_YEAR,
            period_month=PERIOD_MONTH,
            client_id=lha_world["clinic_client"].id,
            institution_patient_id=lha_world["patients"]["cavadini"].id,
            include_line_details=True,
            now=SEP_1,
        )
        patient_ids = {int(pl.booking_id) for pl in patient_preview.preview_lines}
        assert patient_ids == {lha_world["bookings"]["market_validated_patient"].id}
        assert float(patient_preview.estimated_total) == float(
            patient_bucket.estimated_total
        )

        patient_result = GenerateInvoiceUseCase(pdf_service=pdf).execute(
            GenerateInvoiceInput(
                company_id=lha_world["transport"].id,
                client_id=lha_world["clinic_client"].id,
                period_year=PERIOD_YEAR,
                period_month=PERIOD_MONTH,
                billing_party_id=lha_world["patient_bps"]["cavadini"].id,
                reservation_ids=[lha_world["bookings"]["market_validated_patient"].id],
            ),
            now=SEP_1,
        )
        assert patient_result.success is True, patient_result.error
        assert patient_result.invoice_id is not None
        p_ids: set[int] = set()
        p_ht = 0.0
        for line in InvoiceLine.query.filter_by(
            invoice_id=patient_result.invoice_id
        ).all():
            p_ht = round(p_ht + float(line.line_total), 2)
            if line.reservation_id is not None:
                p_ids.add(int(line.reservation_id))
        assert p_ids == {lha_world["bookings"]["market_validated_patient"].id}
        assert p_ht == float(patient_bucket.estimated_total)
