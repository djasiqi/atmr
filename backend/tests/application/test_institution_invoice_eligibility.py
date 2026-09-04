"""Origine × gate Market LIRIE × payeur — source de vérité « Nouvelle facture »."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from application.invoices.institution_invoice_eligibility import (
    ORIGIN_MARKET_LIRIE,
    ORIGIN_OWN_PORTFOLIO,
    filter_institution_invoice_eligible,
    invoice_gate_status,
    is_institution_invoice_eligible,
    is_market_lirie_deadline_passed,
    market_lirie_deadline,
    reopen_market_lirie_validation_after_financial_change,
    resolve_commercial_origin,
    resolve_invoice_payer_type,
)
from application.invoices.invoice_booking_units import resolve_invoice_booking_units
from models.enums import InstitutionBillingControlStatus

ZURICH = ZoneInfo("Europe/Zurich")


def _bk(**kwargs):
    defaults = {
        "id": 1,
        "billing_origin": "OWN_PORTFOLIO",
        "created_via": "company_manual",
        "institution_control_status": None,
        "billed_to_type": "clinic",
        "billed_to_company_id": 10,
        "billing_party_id": 100,
        "scheduled_time": datetime(2026, 8, 15, 10, 0, tzinfo=ZURICH),
        "source_request": None,
        "parent_booking_id": None,
        "route_group_id": None,
        "is_return": False,
        "pickup_location": "LHA",
        "dropoff_location": "HUG",
        "client_id": 1,
        "institution_patient_id": None,
        "amount": 40,
        "_resolve_source_transport_request": lambda: None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_february_deadline_is_last_calendar_day():
    deadline = market_lirie_deadline(datetime(2026, 2, 10, 8, 0, tzinfo=ZURICH))
    assert deadline.day == 28
    assert deadline.month == 2
    assert not is_market_lirie_deadline_passed(
        datetime(2026, 2, 10, tzinfo=ZURICH),
        now=datetime(2026, 2, 28, 23, 59, 59, tzinfo=ZURICH),
    )
    assert is_market_lirie_deadline_passed(
        datetime(2026, 2, 10, tzinfo=ZURICH),
        now=datetime(2026, 3, 1, 0, 0, 0, tzinfo=ZURICH),
    )


def test_august_deadline_releases_on_first_september():
    service = datetime(2026, 8, 5, 10, 0, tzinfo=ZURICH)
    last_instant = datetime(2026, 8, 31, 23, 59, 59, tzinfo=ZURICH)
    first_next = datetime(2026, 9, 1, 0, 0, 0, tzinfo=ZURICH)
    assert not is_market_lirie_deadline_passed(service, now=last_instant)
    assert is_market_lirie_deadline_passed(service, now=first_next)
    pending = _bk(
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.PENDING_REVIEW,
        created_via="institution_portal",
        scheduled_time=service,
    )
    disputed = _bk(
        id=9,
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.ANOMALY,
        created_via="institution_portal",
        scheduled_time=service,
    )
    assert invoice_gate_status(pending, now=last_instant) == "pending"
    assert not is_institution_invoice_eligible(pending, now=last_instant)
    assert invoice_gate_status(pending, now=first_next) == "auto_released"
    assert is_institution_invoice_eligible(pending, now=first_next)
    assert invoice_gate_status(disputed, now=last_instant) == "disputed"
    assert invoice_gate_status(disputed, now=first_next) == "disputed"
    assert not is_institution_invoice_eligible(disputed, now=first_next)


def test_portfolio_has_no_market_gate():
    b = _bk(billing_origin="OWN_PORTFOLIO")
    assert resolve_commercial_origin(b) == ORIGIN_OWN_PORTFOLIO
    assert invoice_gate_status(b) == "not_required"
    assert is_institution_invoice_eligible(b)


def test_market_pending_blocked_during_month():
    b = _bk(
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.PENDING_REVIEW,
        created_via="institution_portal",
    )
    now = datetime(2026, 8, 20, 12, 0, tzinfo=ZURICH)
    assert invoice_gate_status(b, now=now) == "pending"
    assert not is_institution_invoice_eligible(b, now=now)


def test_market_pending_auto_released_after_deadline_not_validated():
    b = _bk(
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.PENDING_REVIEW,
        created_via="institution_portal",
    )
    now = datetime(2026, 9, 4, 8, 0, tzinfo=ZURICH)
    assert invoice_gate_status(b, now=now) == "auto_released"
    assert is_institution_invoice_eligible(b, now=now)
    assert (
        b.institution_control_status is InstitutionBillingControlStatus.PENDING_REVIEW
    )


def test_market_disputed_never_auto_released():
    b = _bk(
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.ANOMALY,
        created_via="institution_portal",
    )
    now = datetime(2026, 9, 4, 8, 0, tzinfo=ZURICH)
    assert invoice_gate_status(b, now=now) == "disputed"
    assert not is_institution_invoice_eligible(b, now=now)


def test_market_validated_is_eligible():
    b = _bk(
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.VALIDATED,
        created_via="institution_portal",
    )
    now = datetime(2026, 8, 20, 12, 0, tzinfo=ZURICH)
    assert invoice_gate_status(b, now=now) == "validated"
    assert is_institution_invoice_eligible(b, now=now)


def test_payer_is_independent_from_origin():
    aller = _bk(
        id=1,
        billing_origin="LIRIE_MARKETPLACE",
        billed_to_type="clinic",
        created_via="institution_portal",
    )
    retour = _bk(
        id=2,
        billing_origin="LIRIE_MARKETPLACE",
        billed_to_type="patient",
        created_via="institution_portal",
        parent_booking_id=1,
        is_return=True,
    )
    assert resolve_invoice_payer_type(aller) == "clinic"
    assert resolve_invoice_payer_type(retour) == "patient"


def test_round_trip_different_payers_stay_separate():
    aller = _bk(
        id=10,
        billed_to_type="clinic",
        billing_party_id=501,
        pickup_location="LHA Anières",
        dropoff_location="HUG Genève",
        institution_patient_id=7,
    )
    retour = _bk(
        id=11,
        billed_to_type="patient",
        billing_party_id=802,
        parent_booking_id=10,
        is_return=True,
        pickup_location="HUG Genève",
        dropoff_location="LHA Anières",
        institution_patient_id=7,
        scheduled_time=datetime(2026, 8, 15, 16, 0, tzinfo=ZURICH),
    )
    units = resolve_invoice_booking_units(
        selected_ids=None,
        scope_bookings=[aller, retour],
    )
    assert len(units) == 2
    assert {u.kind for u in units} == {"single"}


def test_round_trip_same_payer_and_parent_merges():
    aller = _bk(
        id=20,
        billed_to_type="clinic",
        billing_party_id=501,
        pickup_location="LHA Anières",
        dropoff_location="HUG Genève",
        institution_patient_id=8,
    )
    retour = _bk(
        id=21,
        billed_to_type="clinic",
        billing_party_id=501,
        parent_booking_id=20,
        is_return=True,
        pickup_location="HUG Genève",
        dropoff_location="LHA Anières",
        institution_patient_id=8,
        scheduled_time=datetime(2026, 8, 15, 16, 0, tzinfo=ZURICH),
    )
    units = resolve_invoice_booking_units(
        selected_ids=None,
        scope_bookings=[aller, retour],
    )
    assert len(units) == 1
    assert units[0].kind == "round_trip"
    assert set(units[0].booking_ids) == {20, 21}


def test_round_trip_same_request_id_merges():
    aller = _bk(
        id=30,
        billed_to_type="clinic",
        billing_party_id=501,
        pickup_location="LHA Anières",
        dropoff_location="HUG Genève",
        institution_patient_id=9,
        _invoice_request_id=777,
    )
    retour = _bk(
        id=31,
        billed_to_type="clinic",
        billing_party_id=501,
        is_return=True,
        pickup_location="HUG Genève",
        dropoff_location="LHA Anières",
        institution_patient_id=9,
        scheduled_time=datetime(2026, 8, 15, 16, 0, tzinfo=ZURICH),
        _invoice_request_id=777,
    )
    units = resolve_invoice_booking_units(
        selected_ids=None,
        scope_bookings=[aller, retour],
    )
    assert len(units) == 1
    assert units[0].kind == "round_trip"


def test_filter_keeps_portfolio_and_released_excludes_pending_disputed():
    now = datetime(2026, 8, 20, tzinfo=ZURICH)
    portfolio = _bk(id=1, billing_origin="OWN_PORTFOLIO")
    pending = _bk(
        id=2,
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.PENDING_REVIEW,
        created_via="institution_portal",
    )
    disputed = _bk(
        id=3,
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.ANOMALY,
        created_via="institution_portal",
    )
    validated = _bk(
        id=4,
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.VALIDATED,
        created_via="institution_portal",
    )
    kept = filter_institution_invoice_eligible(
        [portfolio, pending, disputed, validated], now=now
    )
    assert {b.id for b in kept} == {1, 4}


def test_financial_change_reopens_validated_market_not_portfolio():
    market = _bk(
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.VALIDATED,
        created_via="institution_portal",
    )
    portfolio = _bk(
        billing_origin="OWN_PORTFOLIO",
        institution_control_status=InstitutionBillingControlStatus.VALIDATED,
    )
    assert reopen_market_lirie_validation_after_financial_change(market) is True
    assert (
        market.institution_control_status
        == InstitutionBillingControlStatus.PENDING_REVIEW
    )
    assert reopen_market_lirie_validation_after_financial_change(portfolio) is False
    assert (
        portfolio.institution_control_status
        == InstitutionBillingControlStatus.VALIDATED
    )


def test_not_billable_after_dispute_never_eligible():
    b = _bk(
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.ANOMALY,
        created_via="institution_portal",
        invoice_billing_status="not_billable",
    )
    assert invoice_gate_status(b) == "not_billable"
    assert not is_institution_invoice_eligible(b)


def test_validated_after_dispute_is_eligible():
    b = _bk(
        billing_origin="LIRIE_MARKETPLACE",
        institution_control_status=InstitutionBillingControlStatus.VALIDATED,
        created_via="institution_portal",
        invoice_billing_status="billable",
    )
    assert invoice_gate_status(b) == "validated_after_dispute"
    assert is_institution_invoice_eligible(b)
