"""Invariant temporel A/R — confirmation retour et invalidation aval."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

from application.bookings.round_trip_temporal import (
    RETURN_BEFORE_APPOINTMENT_MESSAGE,
    RETURN_BEFORE_OUTBOUND_MESSAGE,
    TEMPORAL_CONFLICT_CODE,
    apply_round_trip_schedule_rules,
    evaluate_return_chronology,
    invalidate_impossible_downstream_confirmations,
)
from application.companies.reservations.schedule_reservation import (
    ScheduleCompanyReservationUseCase,
)
from application.companies.reservations.update_reservation import (
    UpdateCompanyReservationUseCase,
)
from models.enums import BookingStatus


def _booking(**kwargs):
    defaults = {
        "id": 1,
        "status": BookingStatus.ACCEPTED.value,
        "is_return": False,
        "parent_booking_id": None,
        "scheduled_time": datetime(2026, 9, 12, 14, 15),
        "time_confirmed": True,
        "pickup_location": "Anières",
        "dropoff_location": "HUG",
        "original_booking": None,
        "return_trip": None,
        "_downstream_returns": None,
        "source_request": None,
        "route_group_id": None,
        "route_sequence_number": 1,
        "_previous_leg": None,
        "edit_version": 1,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestEvaluateReturnChronology:
    def test_return_before_outbound_is_conflict(self):
        conflict = evaluate_return_chronology(
            outbound_pickup=datetime(2026, 9, 12, 14, 15),
            return_pickup=datetime(2026, 9, 12, 12, 15),
        )
        assert conflict is not None
        assert conflict.code == TEMPORAL_CONFLICT_CODE
        assert conflict.message == RETURN_BEFORE_OUTBOUND_MESSAGE

    def test_return_before_appointment_is_conflict(self):
        conflict = evaluate_return_chronology(
            outbound_pickup=datetime(2026, 9, 12, 14, 15),
            return_pickup=datetime(2026, 9, 12, 14, 45),
            appointment_dt=datetime(2026, 9, 12, 15, 0),
        )
        assert conflict is not None
        assert conflict.message == RETURN_BEFORE_APPOINTMENT_MESSAGE

    def test_valid_afternoon_return_stays_ok(self):
        assert (
            evaluate_return_chronology(
                outbound_pickup=datetime(2026, 9, 12, 12, 15),
                return_pickup=datetime(2026, 9, 12, 16, 0),
            )
            is None
        )


class TestInvalidateDownstream:
    def test_impossible_return_loses_confirmation_only(self):
        ret = _booking(
            id=2,
            is_return=True,
            parent_booking_id=1,
            scheduled_time=datetime(2026, 9, 12, 12, 15),
            time_confirmed=True,
            status=BookingStatus.ACCEPTED.value,
        )
        outbound = _booking(
            id=1,
            scheduled_time=datetime(2026, 9, 12, 14, 15),
            _downstream_returns=[ret],
        )
        updated = invalidate_impossible_downstream_confirmations(outbound)
        assert "return[2].time_confirmed" in updated
        assert ret.time_confirmed is False
        assert ret.scheduled_time == datetime(2026, 9, 12, 12, 15)
        assert ret.status == BookingStatus.ACCEPTED.value

    def test_valid_return_stays_confirmed(self):
        ret = _booking(
            id=2,
            is_return=True,
            parent_booking_id=1,
            scheduled_time=datetime(2026, 9, 12, 16, 0),
            time_confirmed=True,
        )
        outbound = _booking(
            id=1,
            scheduled_time=datetime(2026, 9, 12, 12, 15),
            _downstream_returns=[ret],
        )
        updated = invalidate_impossible_downstream_confirmations(outbound)
        assert updated == []
        assert ret.time_confirmed is True

    def test_small_outbound_shift_keeps_later_return(self):
        ret = _booking(
            id=2,
            is_return=True,
            parent_booking_id=1,
            scheduled_time=datetime(2026, 9, 12, 16, 0),
            time_confirmed=True,
        )
        outbound = _booking(
            id=1,
            scheduled_time=datetime(2026, 9, 12, 12, 15),
            _downstream_returns=[ret],
        )
        updated = invalidate_impossible_downstream_confirmations(outbound)
        assert updated == []
        assert ret.time_confirmed is True


class TestUpdateReservationRules:
    def test_reject_confirming_return_before_outbound(self):
        outbound = _booking(id=10, scheduled_time=datetime(2026, 9, 12, 14, 15))
        ret = _booking(
            id=20,
            is_return=True,
            parent_booking_id=10,
            return_trip=outbound,
            scheduled_time=datetime(2026, 9, 12, 16, 0),
            time_confirmed=False,
            amount=50.0,
        )
        result = UpdateCompanyReservationUseCase().execute(
            ret,
            validated_data={
                "scheduled_time": "2026-09-12T12:15:00",
                "time_confirmed": True,
            },
        )
        assert result.ok is False
        assert result.status_code == 422
        assert result.error["error"] == TEMPORAL_CONFLICT_CODE
        assert result.error["message"] == RETURN_BEFORE_OUTBOUND_MESSAGE
        assert ret.time_confirmed is False

    def test_outbound_shift_invalidates_stale_return(self, monkeypatch):
        monkeypatch.setattr(
            "services.institutions.mission_schedule.sync_request_departure_for_booking",
            lambda *_a, **_k: True,
        )
        ret = _booking(
            id=20,
            is_return=True,
            parent_booking_id=10,
            scheduled_time=datetime(2026, 9, 12, 13, 30),
            time_confirmed=True,
            status=BookingStatus.ACCEPTED.value,
        )
        outbound = _booking(
            id=10,
            scheduled_time=datetime(2026, 9, 12, 12, 0),
            time_confirmed=True,
            amount=50.0,
            _downstream_returns=[ret],
        )
        result = UpdateCompanyReservationUseCase().execute(
            outbound,
            validated_data={
                "scheduled_time": "2026-09-12T14:15:00",
                "time_confirmed": True,
            },
        )
        assert result.ok is True
        assert outbound.scheduled_time.hour == 14
        assert ret.time_confirmed is False
        assert ret.status == BookingStatus.ACCEPTED.value

    def test_outbound_small_shift_keeps_valid_return(self, monkeypatch):
        monkeypatch.setattr(
            "services.institutions.mission_schedule.sync_request_departure_for_booking",
            lambda *_a, **_k: True,
        )
        ret = _booking(
            id=20,
            is_return=True,
            parent_booking_id=10,
            scheduled_time=datetime(2026, 9, 12, 16, 0),
            time_confirmed=True,
        )
        outbound = _booking(
            id=10,
            scheduled_time=datetime(2026, 9, 12, 12, 0),
            time_confirmed=True,
            amount=50.0,
            _downstream_returns=[ret],
        )
        result = UpdateCompanyReservationUseCase().execute(
            outbound,
            validated_data={
                "scheduled_time": "2026-09-12T12:15:00",
                "time_confirmed": True,
            },
        )
        assert result.ok is True
        assert ret.time_confirmed is True


class TestScheduleReservationRules:
    def test_schedule_return_before_outbound_rejected(self):
        outbound = _booking(id=10, scheduled_time=datetime(2026, 9, 12, 14, 15))
        ret = _booking(
            id=20,
            is_return=True,
            parent_booking_id=10,
            return_trip=outbound,
            scheduled_time=None,
            time_confirmed=False,
            amount=50.0,
        )
        result = ScheduleCompanyReservationUseCase().execute(
            ret,
            scheduled_time_iso="2026-09-12T12:15:00",
            time_confirmed=True,
        )
        assert result.ok is False
        assert result.status_code == 422
        assert result.error["error"] == TEMPORAL_CONFLICT_CODE


class TestRouteGroupTopology:
    def test_reject_leg2_before_leg1(self):
        outbound = _booking(
            id=45726,
            route_group_id="grp-cavadini",
            route_sequence_number=1,
            scheduled_time=datetime(2026, 9, 12, 14, 15),
        )
        ret = _booking(
            id=45727,
            is_return=False,
            parent_booking_id=None,
            route_group_id="grp-cavadini",
            route_sequence_number=2,
            _previous_leg=outbound,
            scheduled_time=datetime(2026, 9, 12, 16, 0),
            time_confirmed=False,
            amount=50.0,
        )
        result = UpdateCompanyReservationUseCase().execute(
            ret,
            validated_data={
                "scheduled_time": "2026-09-12T12:15:00",
                "time_confirmed": True,
            },
        )
        assert result.ok is False
        assert result.status_code == 422
        assert result.error["error"] == TEMPORAL_CONFLICT_CODE

    def test_outbound_shift_invalidates_route_group_return(self, monkeypatch):
        monkeypatch.setattr(
            "services.institutions.mission_schedule.sync_request_departure_for_booking",
            lambda *_a, **_k: True,
        )
        ret = _booking(
            id=45727,
            is_return=False,
            route_group_id="grp-cavadini",
            route_sequence_number=2,
            scheduled_time=datetime(2026, 9, 12, 12, 15),
            time_confirmed=True,
            status=BookingStatus.ACCEPTED.value,
        )
        outbound = _booking(
            id=45726,
            route_group_id="grp-cavadini",
            route_sequence_number=1,
            scheduled_time=datetime(2026, 9, 12, 12, 15),
            time_confirmed=True,
            amount=50.0,
            _downstream_returns=[ret],
        )
        result = UpdateCompanyReservationUseCase().execute(
            outbound,
            validated_data={
                "scheduled_time": "2026-09-12T14:15:00",
                "time_confirmed": True,
            },
        )
        assert result.ok is True
        assert ret.time_confirmed is False
        assert ret.status == BookingStatus.ACCEPTED.value


class TestApplyRulesWithoutMutation:
    def test_unconfirmed_return_is_not_rejected(self):
        outbound = _booking(id=10, scheduled_time=datetime(2026, 9, 12, 14, 15))
        ret = _booking(
            id=20,
            is_return=True,
            parent_booking_id=10,
            return_trip=outbound,
        )
        assert (
            apply_round_trip_schedule_rules(
                ret,
                intended_pickup=datetime(2026, 9, 12, 12, 15),
                intended_confirmed=False,
            )
            is None
        )
