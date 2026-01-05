from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from domain.events.events import DispatchRequestedEvent


class _CompanyLike(Protocol):
    id: int | None
    dispatch_enabled: bool


class _CompanyRepo(Protocol):
    def find_model_by_id(self, company_id: int) -> _CompanyLike | None: ...


@dataclass(frozen=True, slots=True)
class RequestDispatchCommand:
    company_id: int
    action: str = "update"
    reason: str | None = None


class RequestDispatchUseCase:
    """Use-case Application: demander un dispatch (via event bus).

    Règle: publier `DispatchRequestedEvent` uniquement si `dispatch_enabled=True`.
    """

    def __init__(
        self,
        *,
        company_repo: _CompanyRepo,
        publish_event_fn: Callable[[DispatchRequestedEvent], None],
    ) -> None:
        super().__init__()
        self._company_repo = company_repo
        self._publish_event = publish_event_fn

    def execute(self, cmd: RequestDispatchCommand) -> None:
        company = self._company_repo.find_model_by_id(cmd.company_id)
        if not company or not bool(getattr(company, "dispatch_enabled", False)):
            return

        self._publish_event(
            DispatchRequestedEvent(
                company_id=cmd.company_id,
                action=cmd.action,
                reason=cmd.reason,
            )
        )

    def execute_for_driver_change(self, company: _CompanyLike, *, action: str) -> None:
        if not company or not bool(getattr(company, "dispatch_enabled", False)):
            return
        company_id_obj = getattr(company, "id", None)
        if company_id_obj is None:
            return
        try:
            company_id = int(company_id_obj)
        except Exception:
            return

        self._publish_event(
            DispatchRequestedEvent(
                company_id=company_id,
                action=action,
                reason=f"driver_{action}",
            )
        )
