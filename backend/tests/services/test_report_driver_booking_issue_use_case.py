from __future__ import annotations

from dataclasses import dataclass

from application.drivers.report_driver_booking_issue import (
    ReportDriverBookingIssueUseCase,
)


@dataclass
class _Booking:
    issue_report: str | None = None


class _Repo:
    def __init__(self, booking: _Booking | None):
        self._booking = booking

    def find_model_by_id_and_driver(self, booking_id: int, driver_id: int):  # type: ignore[no-untyped-def]
        _ = (booking_id, driver_id)
        return self._booking


def test_report_issue_returns_404_when_booking_missing() -> None:
    uc = ReportDriverBookingIssueUseCase(booking_repo=_Repo(None))
    res = uc.execute(booking_id=1, driver_id=2, payload={"issue": "x"})
    assert res.status_code == 404


def test_report_issue_returns_400_when_issue_missing() -> None:
    booking = _Booking()
    uc = ReportDriverBookingIssueUseCase(booking_repo=_Repo(booking))
    res = uc.execute(booking_id=1, driver_id=2, payload={})
    assert res.status_code == 400
    assert booking.issue_report is None


def test_report_issue_sets_issue_report() -> None:
    booking = _Booking()
    uc = ReportDriverBookingIssueUseCase(booking_repo=_Repo(booking))
    res = uc.execute(booking_id=1, driver_id=2, payload={"issue": "bad gps"})
    assert res.status_code == 200
    assert booking.issue_report == "bad gps"
