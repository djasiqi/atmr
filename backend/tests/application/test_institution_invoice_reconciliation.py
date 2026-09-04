"""Conservation du registre de réconciliation institution."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from application.invoices.institution_invoice_reconciliation import (
    build_reconciliation_ledger,
)
from models.enums import InstitutionBillingControlStatus

ZURICH = ZoneInfo("Europe/Zurich")


def _bk(**kwargs):
    defaults = {
        "id": 1,
        "billing_origin": "LIRIE_MARKETPLACE",
        "created_via": "institution_portal",
        "institution_control_status": InstitutionBillingControlStatus.VALIDATED,
        "billed_to_type": "clinic",
        "billing_party_id": 1,
        "invoice_line_id": None,
        "scheduled_time": datetime(2026, 8, 10, 10, 0, tzinfo=ZURICH),
        "amount": 40,
        "parent_booking_id": None,
        "route_group_id": None,
        "source_request": None,
        "_resolve_source_transport_request": lambda: None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_conservation_no_duplicate_no_disappearance():
    now = datetime(2026, 8, 31, 23, 59, 59, tzinfo=ZURICH)
    bookings = [
        _bk(id=1, billed_to_type="clinic"),
        _bk(
            id=2,
            billed_to_type="patient",
            institution_control_status=InstitutionBillingControlStatus.PENDING_REVIEW,
        ),
        _bk(
            id=3,
            billed_to_type="clinic",
            institution_control_status=InstitutionBillingControlStatus.ANOMALY,
        ),
        _bk(id=4, billed_to_type="clinic", invoice_line_id=99),
    ]
    ledger = build_reconciliation_ledger(
        bookings, period_year=2026, period_month=8, now=now
    )
    assert ledger.conservation_ok is True
    assert ledger.considered_count == 4
    assert ledger.considered_amount_ht == 160.0
    d = ledger.to_dict()
    assert set(d["buckets"]["clinic_billable"]["booking_ids"]) == {1}
    assert set(d["buckets"]["pending_blocked"]["booking_ids"]) == {2}
    assert set(d["buckets"]["disputed_blocked"]["booking_ids"]) == {3}
    assert set(d["buckets"]["already_invoiced"]["booking_ids"]) == {4}
