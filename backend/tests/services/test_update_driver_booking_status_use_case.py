from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from application.drivers.update_driver_booking_status import (
    UpdateDriverBookingStatusCommand,
    UpdateDriverBookingStatusUseCase,
)
from models import BookingStatus
from models.enums import CancelReason


@dataclass
class _Booking:
    id: int
    company_id: int
    driver_id: int | None
    status: BookingStatus
    is_return: bool = False
    boarded_at: datetime | None = None
    completed_at: datetime | None = None


class _BookingRepo:
    def __init__(self, booking: _Booking | None):
        self._booking = booking

    def find_model_by_id(self, booking_id: int):  # type: ignore[no-untyped-def]
        if self._booking is None or self._booking.id != booking_id:
            return None
        return self._booking


@dataclass
class _Assignment:
    id: int


class _AssignmentRepo:
    def __init__(self, assignment: _Assignment | None):
        self._assignment = assignment

    def find_model_by_booking_id(self, booking_id: int):  # type: ignore[no-untyped-def]
        _ = booking_id
        return self._assignment


class _Db:
    def __init__(self) -> None:
        self.commits = 0
        self.deleted: list[object] = []

    def commit(self) -> None:
        self.commits += 1

    def delete(self, obj: object) -> None:
        self.deleted.append(obj)


def test_en_route_requires_assigned() -> None:
    booking = _Booking(id=1, company_id=1, driver_id=10, status=BookingStatus.PENDING)
    db = _Db()
    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(None),
        db_session=db,
        notify_booking_update_fn=lambda _driver_id, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _cid, _aid, _bid, _did: None,
        maybe_trigger_dispatch_fn=None,
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )
    res = uc.execute(
        UpdateDriverBookingStatusCommand(
            booking_id=1,
            driver_id=10,
            payload={"status": "en_route"},
        )
    )
    assert res.status_code == 400
    assert db.commits == 0


def test_release_cancels_assignment_and_triggers_dispatch() -> None:
    from unittest.mock import patch

    booking = _Booking(id=1, company_id=7, driver_id=10, status=BookingStatus.ASSIGNED)
    assignment = _Assignment(id=123)
    db = _Db()
    events: dict[str, Any] = {"emit": 0, "trigger": 0}

    uc = UpdateDriverBookingStatusUseCase(
        booking_repo=_BookingRepo(booking),
        assignment_repo=_AssignmentRepo(assignment),
        db_session=db,
        notify_booking_update_fn=lambda _driver_id, _b: None,
        resolve_delays_fn=lambda _bid, _dt: None,
        emit_assignment_cancelled_fn=lambda _cid, _aid, _bid, _did: events.__setitem__(
            "emit", events["emit"] + 1
        ),
        maybe_trigger_dispatch_fn=lambda _cid, _action: events.__setitem__(
            "trigger", events["trigger"] + 1
        ),
        now_utc_fn=lambda: datetime(2025, 12, 12, 10, 0, 0, tzinfo=UTC),
    )

    # Mock publish_event pour qu'il échoue, forçant l'appel à emit_assignment_cancelled
    with patch(
        "application.events.event_bus.publish_event",
        side_effect=Exception("Event publish failed"),
    ):
        res = uc.execute(
            UpdateDriverBookingStatusCommand(
                booking_id=1,
                driver_id=10,
                payload={
                    "status": "canceled",
                    "cancel_reason": CancelReason.RELEASE.value,
                },
            )
        )
    assert res.status_code == 200
    assert booking.status == BookingStatus.ACCEPTED
    assert booking.driver_id is None
    assert db.deleted  # assignment deleted
    assert events["emit"] == 1
    assert events["trigger"] == 1
