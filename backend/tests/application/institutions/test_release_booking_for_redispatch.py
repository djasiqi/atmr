"""Couverture de ``ReleaseBookingForRedispatchUseCase``."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from application.institutions.release_booking_for_redispatch import (
    ReleaseBookingForRedispatchInput,
    ReleaseBookingForRedispatchUseCase,
)
from models.enums import BookingStatus

_MOD = "application.institutions.release_booking_for_redispatch"


def _booking(**kwargs):
    defaults = {
        "id": 1,
        "status": BookingStatus.ASSIGNED,
        "company_id": 10,
        "executing_company_id": 11,
        "driver_id": 5,
        "active_change_request_id": 99,
        "updated_at": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _query(*, get=None, first=None, filter_side_effect=None):
    q = MagicMock()
    q.get.return_value = get
    if filter_side_effect is not None:
        q.filter_by.side_effect = filter_side_effect
    else:
        q.filter_by.return_value.first.return_value = first
    return SimpleNamespace(query=q)


def _patch_booking(monkeypatch, booking):
    model = MagicMock()
    model.query.get.return_value = booking
    monkeypatch.setattr(f"{_MOD}.Booking", model)
    monkeypatch.setattr(f"{_MOD}.db", MagicMock())
    return model


def _silent_side_effects(monkeypatch, *, offers: int = 1, redisp_ok: bool = True):
    monkeypatch.setattr(
        "models.Company",
        _query(get=SimpleNamespace(name="Taxi")),
    )
    monkeypatch.setattr(
        "models.TransportRequest",
        _query(first=SimpleNamespace(id=44)),
    )
    monkeypatch.setattr(
        "services.institutions.transport_timeline_service.record_event",
        MagicMock(),
    )

    class _FakeRedispatch:
        def execute(self, _inp):
            return SimpleNamespace(success=redisp_ok, offers_created=offers)

    monkeypatch.setattr(
        "application.institutions.redispatch_institution_booking.RedispatchInstitutionBookingUseCase",
        _FakeRedispatch,
    )
    audit = MagicMock()
    monkeypatch.setattr(f"{_MOD}.AuditLogger", SimpleNamespace(log_action=audit))
    return audit


def test_404_course_introuvable(monkeypatch):
    _patch_booking(monkeypatch, None)
    result = ReleaseBookingForRedispatchUseCase().execute(
        ReleaseBookingForRedispatchInput(booking_id=99)
    )
    assert result.success is False
    assert result.status_code == 404
    assert "introuvable" in (result.error or "")


@pytest.mark.parametrize(
    "status",
    [
        BookingStatus.COMPLETED,
        BookingStatus.RETURN_COMPLETED,
        "CANCELED",
    ],
)
def test_409_statut_terminal(monkeypatch, status):
    _patch_booking(monkeypatch, _booking(status=status))
    result = ReleaseBookingForRedispatchUseCase().execute(
        ReleaseBookingForRedispatchInput(booking_id=1)
    )
    assert result.success is False
    assert result.status_code == 409
    assert "Libération impossible" in (result.error or "")


def test_succes_liberation_et_redispatch(monkeypatch):
    booking = _booking()
    _patch_booking(monkeypatch, booking)
    audit = _silent_side_effects(monkeypatch, offers=3)
    result = ReleaseBookingForRedispatchUseCase().execute(
        ReleaseBookingForRedispatchInput(
            booking_id=1,
            institution_id=2,
            reason="refus",
            previous_company_id=8,
            actor_user_id=7,
        )
    )
    assert result.success is True
    assert result.redispatched is True
    assert result.offers_created == 3
    assert result.previous_company_id == 8
    assert booking.driver_id is None
    assert booking.company_id is None
    assert booking.executing_company_id is None
    assert booking.active_change_request_id is None
    assert booking.status == BookingStatus.PENDING
    audit.assert_called_once()


def test_company_id_executing_et_lookup_none(monkeypatch):
    booking = _booking(company_id=None, executing_company_id=15)
    _patch_booking(monkeypatch, booking)
    company = _query(get=None)
    monkeypatch.setattr("models.Company", company)
    monkeypatch.setattr(
        "services.institutions.transport_timeline_service.record_event",
        MagicMock(),
    )
    monkeypatch.setattr("models.TransportRequest", _query(first=None))
    monkeypatch.setattr(f"{_MOD}.AuditLogger", SimpleNamespace(log_action=MagicMock()))
    result = ReleaseBookingForRedispatchUseCase().execute(
        ReleaseBookingForRedispatchInput(booking_id=1, trigger_redispatch=False)
    )
    assert result.success is True
    assert result.redispatched is False
    assert result.previous_company_id == 15
    company.query.get.assert_called_once_with(15)


def test_company_lookup_exception(monkeypatch):
    booking = _booking()
    _patch_booking(monkeypatch, booking)
    company = _query()
    company.query.get.side_effect = RuntimeError("db")
    monkeypatch.setattr("models.Company", company)
    monkeypatch.setattr(
        "services.institutions.transport_timeline_service.record_event",
        MagicMock(),
    )
    monkeypatch.setattr("models.TransportRequest", _query(first=None))
    monkeypatch.setattr(f"{_MOD}.AuditLogger", SimpleNamespace(log_action=MagicMock()))
    result = ReleaseBookingForRedispatchUseCase().execute(
        ReleaseBookingForRedispatchInput(booking_id=1, trigger_redispatch=False)
    )
    assert result.success is True
    assert result.previous_company_id == 10


def test_redispatch_et_audit_en_echec(monkeypatch, caplog):
    booking = _booking(company_id=None, executing_company_id=None)
    _patch_booking(monkeypatch, booking)
    monkeypatch.setattr(
        "services.institutions.transport_timeline_service.record_event",
        MagicMock(),
    )
    monkeypatch.setattr("models.TransportRequest", _query(first=None))

    class _BoomRedispatch:
        def execute(self, _inp):
            raise RuntimeError("redispatch down")

    monkeypatch.setattr(
        "application.institutions.redispatch_institution_booking.RedispatchInstitutionBookingUseCase",
        _BoomRedispatch,
    )
    monkeypatch.setattr(
        f"{_MOD}.AuditLogger",
        SimpleNamespace(log_action=MagicMock(side_effect=RuntimeError("audit down"))),
    )
    with caplog.at_level("WARNING"):
        result = ReleaseBookingForRedispatchUseCase().execute(
            ReleaseBookingForRedispatchInput(booking_id=1)
        )
    assert result.success is True
    assert result.redispatched is False
    assert result.previous_company_id is None
    assert "redispatch failed" in caplog.text
    assert "audit failed" in caplog.text


def test_timeline_tr_introuvable_et_exceptions(monkeypatch, caplog):
    booking = _booking()
    monkeypatch.setattr("models.TransportRequest", _query(first=None))
    record = MagicMock()
    monkeypatch.setattr(
        "services.institutions.transport_timeline_service.record_event",
        record,
    )
    ReleaseBookingForRedispatchUseCase._record_redispatch_timeline(
        booking=booking,
        institution_id=1,
        previous_company_id=2,
        previous_company_name="Taxi",
        actor_user_id=3,
        reason="x",
    )
    assert record.call_args.kwargs["transport_request_id"] is None

    monkeypatch.setattr(
        "models.TransportRequest",
        _query(filter_side_effect=RuntimeError("tr down")),
    )
    record.reset_mock()
    ReleaseBookingForRedispatchUseCase._record_redispatch_timeline(
        booking=booking,
        institution_id=1,
        previous_company_id=2,
        previous_company_name=None,
        actor_user_id=None,
        reason=None,
    )
    assert record.call_args.kwargs["transport_request_id"] is None

    monkeypatch.setattr(
        "services.institutions.transport_timeline_service.record_event",
        MagicMock(side_effect=RuntimeError("timeline down")),
    )
    with caplog.at_level("WARNING"):
        ReleaseBookingForRedispatchUseCase._record_redispatch_timeline(
            booking=booking,
            institution_id=None,
            previous_company_id=None,
            previous_company_name=None,
            actor_user_id=None,
            reason=None,
        )
    assert "redispatch timeline failed" in caplog.text
