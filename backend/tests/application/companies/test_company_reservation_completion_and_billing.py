"""Tests use cases complétion entreprise + ajustement facturation."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from application.companies.reservations.billing_adjustment import (
    CompanyBookingBillingAdjustmentUseCase,
)
from application.companies.reservations.complete_reservation import (
    CompleteCompanyReservationUseCase,
)
from models.enums import BookingCreatedVia


def _uc_no_transfers(monkeypatch):
    monkeypatch.setattr(
        CompleteCompanyReservationUseCase,
        "_auto_validate_transfers",
        lambda self, booking: None,
    )
    return CompleteCompanyReservationUseCase()


def test_complete_accepted_ok(monkeypatch):
    uc = _uc_no_transfers(monkeypatch)
    b = MagicMock()
    b.status = "accepted"
    b.is_return = False
    b.completed_at = None
    r = uc.execute(b, reason=None)
    assert r.ok is True
    assert r.from_en_route_manual is False


def test_complete_assigned_ok(monkeypatch):
    uc = _uc_no_transfers(monkeypatch)
    b = MagicMock()
    b.status = "assigned"
    b.is_return = False
    b.completed_at = None
    r = uc.execute(b, reason=None)
    assert r.ok is True
    assert r.from_en_route_manual is False


def test_complete_in_progress_ok(monkeypatch):
    uc = _uc_no_transfers(monkeypatch)
    b = MagicMock()
    b.status = "in_progress"
    b.is_return = False
    b.completed_at = None
    r = uc.execute(b, reason=None)
    assert r.ok is True
    assert r.from_en_route_manual is False


def test_complete_rejects_pending():
    uc = CompleteCompanyReservationUseCase()
    b = MagicMock()
    b.status = "pending"
    r = uc.execute(b, reason=None)
    assert r.ok is False
    assert r.status_code == 400


def test_complete_en_route_requires_reason():
    uc = CompleteCompanyReservationUseCase()
    b = MagicMock()
    b.status = "en_route"
    b.is_return = False
    b.completed_at = None
    r = uc.execute(b, reason="  ")
    assert r.ok is False
    assert r.status_code == 400


def test_complete_en_route_ok(monkeypatch):
    monkeypatch.setattr(
        CompleteCompanyReservationUseCase,
        "_auto_validate_transfers",
        lambda self, booking: None,
    )
    uc = CompleteCompanyReservationUseCase()
    b = MagicMock()
    b.status = "en_route"
    b.is_return = False
    b.completed_at = None
    r = uc.execute(b, reason="Chauffeur bloqué")
    assert r.ok is True
    assert r.from_en_route_manual is True


@patch(
    "application.companies.reservations.billing_adjustment._active_invoice_line_exists",
    return_value=False,
)
def test_billing_adjustment_rejects_public_guest(_mock_line):
    uc = CompanyBookingBillingAdjustmentUseCase()
    b = MagicMock()
    b.status = "completed"
    b.created_via = BookingCreatedVia.PUBLIC_GUEST
    b.amount = 40.0
    b.billed_to_type = "patient"
    b.billed_to_company_id = None
    b.id = 1
    b.billing_locked_at = None
    b.invoice_line_id = None
    r = uc.execute(
        b,
        data={"override_reason": "x", "amount": 42.0},
        keys_present={"override_reason", "amount"},
    )
    assert r.ok is False
    assert r.status_code == 400


@patch(
    "application.companies.reservations.billing_adjustment._active_invoice_line_exists",
    return_value=False,
)
def test_billing_adjustment_requires_reason(_mock_line):
    uc = CompanyBookingBillingAdjustmentUseCase()
    b = MagicMock()
    b.status = "completed"
    b.created_via = BookingCreatedVia.DISPATCHER
    b.amount = 40.0
    b.billed_to_type = "patient"
    b.billed_to_company_id = None
    b.id = 1
    b.billing_locked_at = None
    b.invoice_line_id = None
    r = uc.execute(
        b,
        data={"override_reason": ""},
        keys_present={"override_reason"},
    )
    assert r.ok is False


@patch(
    "application.companies.reservations.billing_adjustment._active_invoice_line_exists",
    return_value=False,
)
def test_billing_patient_rejects_extra_company_id(_mock_line):
    uc = CompanyBookingBillingAdjustmentUseCase()
    b = MagicMock()
    b.status = "completed"
    b.created_via = BookingCreatedVia.DISPATCHER
    b.amount = 40.0
    b.billed_to_type = "patient"
    b.billed_to_company_id = None
    b.id = 1
    b.billing_locked_at = None
    b.invoice_line_id = None
    r = uc.execute(
        b,
        data={
            "override_reason": "x",
            "amount": 42.0,
            "billed_to_type": "patient",
            "billed_to_company_id": 99,
        },
        keys_present={
            "override_reason",
            "amount",
            "billed_to_type",
            "billed_to_company_id",
        },
    )
    assert r.ok is False
    assert r.status_code == 400
