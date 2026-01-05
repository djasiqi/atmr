from __future__ import annotations

from dataclasses import dataclass

from application.companies.request_dispatch import (
    RequestDispatchCommand,
    RequestDispatchUseCase,
)
from domain.events.events import DispatchRequestedEvent


@dataclass
class _Company:
    id: int | None
    dispatch_enabled: bool


class _Repo:
    def __init__(self, company: _Company | None):
        self._company = company

    def find_model_by_id(self, _company_id: int) -> _Company | None:
        return self._company


def test_request_dispatch_use_case_does_nothing_if_disabled() -> None:
    events: list[DispatchRequestedEvent] = []

    uc = RequestDispatchUseCase(
        company_repo=_Repo(_Company(id=1, dispatch_enabled=False)),
        publish_event_fn=lambda e: events.append(e),
    )
    uc.execute(RequestDispatchCommand(company_id=1, action="update", reason="x"))
    assert events == []


def test_request_dispatch_use_case_publishes_event_if_enabled() -> None:
    events: list[DispatchRequestedEvent] = []

    uc = RequestDispatchUseCase(
        company_repo=_Repo(_Company(id=1, dispatch_enabled=True)),
        publish_event_fn=lambda e: events.append(e),
    )
    uc.execute(
        RequestDispatchCommand(company_id=1, action="update", reason="booking_update")
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == "DispatchRequestedEvent"
    assert ev.company_id == 1
    assert ev.action == "update"
    assert ev.reason == "booking_update"
