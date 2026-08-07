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
def test_billing_adjustment_allows_institution_portal(_mock_line):
    uc = CompanyBookingBillingAdjustmentUseCase()
    b = MagicMock()
    b.status = "completed"
    b.created_via = BookingCreatedVia.INSTITUTION_PORTAL
    b.amount = 40.0
    b.billed_to_type = "clinic"
    b.billed_to_company_id = 10
    b.billing_party_id = 99
    b.id = 1
    b.is_return = False
    b.company_id = 5
    b.client_id = None
    b.client = None
    b.billing_locked_at = None
    b.invoice_line_id = None
    b._resolve_source_transport_request = MagicMock(return_value=None)
    with patch(
        "application.companies.reservations.billing_adjustment._propagate_payer_to_return_legs",
        return_value=[],
    ), patch(
        "application.companies.reservations.billing_adjustment._apply_billing_party_resolution",
    ):
        r = uc.execute(
            b,
            data={
                "override_reason": "Patient prend en charge",
                "billed_to_type": "patient",
                "billed_to_company_id": None,
            },
            keys_present={
                "override_reason",
                "billed_to_type",
                "billed_to_company_id",
            },
        )
    assert r.ok is True
    assert b.billed_to_type == "patient"
    assert b.billed_to_company_id is None
    assert b.billing_override_reason == "Patient prend en charge"


@patch(
    "application.companies.reservations.billing_adjustment._active_invoice_line_exists",
    return_value=False,
)
def test_billing_adjustment_outbound_propagates_return(_mock_line):
    uc = CompanyBookingBillingAdjustmentUseCase()
    outbound = MagicMock()
    outbound.status = "completed"
    outbound.created_via = BookingCreatedVia.INSTITUTION_PORTAL
    outbound.amount = 40.0
    outbound.billed_to_type = "clinic"
    outbound.billed_to_company_id = 10
    outbound.billing_party_id = 99
    outbound.id = 100
    outbound.is_return = False
    outbound.company_id = 5
    outbound.billing_locked_at = None
    outbound.invoice_line_id = None
    outbound._resolve_source_transport_request = MagicMock(return_value=None)

    with patch(
        "application.companies.reservations.billing_adjustment._propagate_payer_to_return_legs",
        return_value=[101],
    ) as prop, patch(
        "application.companies.reservations.billing_adjustment._apply_billing_party_resolution",
    ):
        r = uc.execute(
            outbound,
            data={
                "override_reason": "Basculer patient",
                "billed_to_type": "patient",
                "billed_to_company_id": None,
            },
            keys_present={
                "override_reason",
                "billed_to_type",
                "billed_to_company_id",
            },
        )
    assert r.ok is True
    assert r.propagated_return_ids == [101]
    prop.assert_called_once()


@patch(
    "application.companies.reservations.billing_adjustment._active_invoice_line_exists",
    return_value=False,
)
def test_billing_adjustment_return_does_not_propagate(_mock_line):
    uc = CompanyBookingBillingAdjustmentUseCase()
    ret = MagicMock()
    ret.status = "completed"
    ret.created_via = BookingCreatedVia.INSTITUTION_PORTAL
    ret.amount = 40.0
    ret.billed_to_type = "clinic"
    ret.billed_to_company_id = 10
    ret.billing_party_id = 99
    ret.id = 101
    ret.is_return = True
    ret.parent_booking_id = 100
    ret.company_id = 5
    ret.billing_locked_at = None
    ret.invoice_line_id = None
    ret._resolve_source_transport_request = MagicMock(return_value=None)

    with patch(
        "application.companies.reservations.billing_adjustment._propagate_payer_to_return_legs",
        return_value=[],
    ) as prop, patch(
        "application.companies.reservations.billing_adjustment._apply_billing_party_resolution",
    ):
        r = uc.execute(
            ret,
            data={
                "override_reason": "Retour seul patient",
                "billed_to_type": "patient",
                "billed_to_company_id": None,
            },
            keys_present={
                "override_reason",
                "billed_to_type",
                "billed_to_company_id",
            },
        )
    assert r.ok is True
    assert r.propagated_return_ids == []
    prop.assert_called_once()


def test_propagate_payer_skips_when_is_return():
    from application.companies.reservations.billing_adjustment import (
        _propagate_payer_to_return_legs,
    )

    ret = MagicMock()
    ret.is_return = True
    ret.id = 2
    assert (
        _propagate_payer_to_return_legs(
            ret, reason="x", terminal_exclude=frozenset({"canceled"})
        )
        == []
    )


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
