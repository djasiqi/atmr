"""Couverture de ``bookings.application.use_cases.cancel_booking``."""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

from bookings.application.use_cases.cancel_booking import (
    CancelBookingUseCase,
    _set_status,
    _status_value,
)


class _Status(str, Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    CANCELED = "canceled"


def test_status_value_variants():
    assert _status_value(None) == ""
    assert _status_value(_Status.PENDING) == "PENDING"
    assert _status_value("assigned") == "assigned"
    assert _status_value(42) == "42"


def test_set_status_enum_et_string():
    booking = SimpleNamespace(status=_Status.PENDING)
    _set_status(booking, "canceled")
    assert booking.status is _Status.CANCELED

    plain = SimpleNamespace(status="pending")
    _set_status(plain, "canceled")
    assert plain.status == "canceled"


def test_cancel_refuse_si_statut_invalide():
    uc = CancelBookingUseCase()
    out = uc.execute(SimpleNamespace(status=_Status.CANCELED, company_id=1))
    assert out.ok is False
    assert out.status_code == 400
    assert out.error is not None

    empty = uc.execute(SimpleNamespace(status=None, company_id=1))
    assert empty.ok is False


def test_cancel_pending_et_assigned():
    uc = CancelBookingUseCase()
    pending = SimpleNamespace(status="Pending", company_id=7)
    out = uc.execute(pending)
    assert out.ok is True
    assert out.company_id == 7
    assert out.should_trigger_dispatch is True
    assert pending.status == "canceled"

    assigned = SimpleNamespace(status=_Status.ASSIGNED, company_id=None)
    out2 = uc.execute(assigned)
    assert out2.ok is True
    assert out2.company_id is None
    assert out2.should_trigger_dispatch is False
    assert assigned.status is _Status.CANCELED


def test_cancel_company_id_invalide():
    uc = CancelBookingUseCase()
    booking = SimpleNamespace(status="assigned", company_id="x")
    out = uc.execute(booking)
    assert out.ok is True
    assert out.company_id is None
    assert out.should_trigger_dispatch is False
