from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infrastructure.dispatch.apply_adapter import apply_assignments
from infrastructure.dispatch.data_adapter import build_problem_data
from infrastructure.dispatch.heuristics_adapter import assign_urgent


@dataclass(frozen=True, slots=True)
class DispatchNowProblemBuilderAdapter:
    def build_problem_data(
        self,
        *,
        company_id: int,
        settings: Any,
        today_str: str,
    ) -> dict[str, Any]:
        # `build_problem_data` retourne un dict
        return build_problem_data(
            company_id=company_id, settings=settings, for_date=today_str
        )


@dataclass(frozen=True, slots=True)
class DispatchNowUrgentAssignerAdapter:
    def assign_urgent(
        self,
        *,
        problem: dict[str, Any],
        urgent_booking_ids: list[int],
        settings: Any,
    ) -> Any:
        return assign_urgent(
            problem=problem, urgent_booking_ids=urgent_booking_ids, settings=settings
        )


@dataclass(frozen=True, slots=True)
class DispatchNowAssignmentsApplierAdapter:
    def apply_assignments(
        self,
        *,
        company_id: int,
        assignments: Any,
        allow_reassign: bool,
        respect_existing: bool,
    ) -> dict[str, Any]:
        return apply_assignments(
            company_id=company_id,
            assignments=assignments,
            allow_reassign=allow_reassign,
            respect_existing=respect_existing,
        )
