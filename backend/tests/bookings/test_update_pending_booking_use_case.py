"""Couverture de ``bookings.application.use_cases.update_pending_booking``."""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

from bookings.application.use_cases.update_pending_booking import (
    UpdatePendingBookingUseCase,
    _set_status,
    _status_value,
)


class _Status(str, Enum):
    PENDING = "pending"
    ASSIGNED = "ASSIGNED"


def _booking(**kwargs):
    data = {
        "id": 1,
        "status": "pending",
        "pickup_location": "A",
        "dropoff_location": "B",
        "scheduled_time": "t0",
        "amount": 10,
        "medical_facility": "hop",
        "doctor_name": "Dr X",
        "notes_medical": "n",
        "pickup_lat": 1.0,
        "pickup_lon": 2.0,
        "dropoff_lat": 3.0,
        "dropoff_lon": 4.0,
    }
    data.update(kwargs)
    return SimpleNamespace(**data)


def test_status_value_et_set_status():
    assert _status_value(None) == ""
    assert _status_value(_Status.PENDING) == "pending"
    assert _status_value("pending") == "pending"
    assert _status_value(7) == "7"

    enum_booking = SimpleNamespace(status=_Status.ASSIGNED)
    _set_status(enum_booking, "pending")
    assert enum_booking.status is _Status.PENDING

    plain = SimpleNamespace(status="assigned")
    _set_status(plain, "pending")
    assert plain.status == "pending"


def test_refuse_si_pas_pending():
    uc = UpdatePendingBookingUseCase()
    out = uc.execute(_booking(status="ASSIGNED"), validated_data={})
    assert out.ok is False
    assert out.status_code == 400

    out2 = uc.execute(_booking(status=None), validated_data={})
    assert out2.ok is False


def test_update_champs_et_adresses():
    uc = UpdatePendingBookingUseCase()
    booking = _booking()
    out = uc.execute(
        booking,
        validated_data={
            "pickup_location": "A2",
            "dropoff_location": "B2",
            "scheduled_time": "t1",
            "amount": 20,
            "medical_facility": "clinique",
            "doctor_name": "Dr Y",
            "notes_medical": "ok",
            "status": "assigned",
        },
    )
    assert out.ok is True
    assert out.addresses_changed is True
    assert out.old_pickup == "A"
    assert out.old_dropoff == "B"
    assert out.new_pickup == "A2"
    assert out.new_dropoff == "B2"
    assert booking.scheduled_time == "t1"
    assert booking.amount == 20
    assert booking.medical_facility == "clinique"
    assert booking.doctor_name == "Dr Y"
    assert booking.notes_medical == "ok"
    assert booking.status == "pending"


def test_adresses_identiques_et_enum_pending():
    uc = UpdatePendingBookingUseCase()
    booking = _booking(status=_Status.PENDING)
    out = uc.execute(
        booking,
        validated_data={
            "pickup_location": "A",
            "dropoff_location": "B",
        },
    )
    assert out.ok is True
    assert out.addresses_changed is False
    assert booking.status is _Status.PENDING
