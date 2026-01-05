from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _ProblemDataBuilderPort(Protocol):
    def build_problem_data(
        self, *, company_id: int, settings: Any, today_str: str
    ) -> dict[str, Any]: ...


class _UrgentAssignerPort(Protocol):
    def assign_urgent(
        self, *, problem: dict[str, Any], urgent_booking_ids: list[int], settings: Any
    ) -> Any: ...


class _AssignmentsApplierPort(Protocol):
    def apply_assignments(
        self,
        *,
        company_id: int,
        assignments: Any,
        allow_reassign: bool,
        respect_existing: bool,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class DispatchNowResult:
    ok: bool
    assigned_driver_id: int | None = None
    error: str | None = None
    should_fallback_trigger_dispatch: bool = False
    fallback_reason: str | None = None


class DispatchNowUseCase:
    """Use-case Application: orchestration 'dispatch-now' (urgent).

    Objectif:
    - Construire un problème dispatch minimal
    - Calculer une assignation urgente
    - Appliquer l'assignation

    La route reste responsable de:
    - contrôles d'accès / statut / horaires (outbound complété, etc.)
    - commit/rollback
    - fallback: déclenchement dispatch classique si nécessaire
    """

    def __init__(
        self,
        *,
        builder: _ProblemDataBuilderPort,
        assigner: _UrgentAssignerPort,
        applier: _AssignmentsApplierPort,
    ) -> None:
        super().__init__()
        self._builder = builder
        self._assigner = assigner
        self._applier = applier

    def execute(
        self,
        *,
        company_id: int,
        booking_id: int,
        today_str: str,
        settings: Any,
    ) -> DispatchNowResult:
        try:
            problem = self._builder.build_problem_data(
                company_id=company_id,
                settings=settings,
                today_str=today_str,
            )
        except Exception as e:
            return DispatchNowResult(
                ok=False,
                error=str(e),
                should_fallback_trigger_dispatch=True,
                fallback_reason="build_problem_error",
            )

        # Même logique que l'existant: fallback si problème incomplet
        bookings = problem.get("bookings", [])
        drivers = problem.get("drivers", [])
        if not bookings or not drivers:
            return DispatchNowResult(
                ok=True,
                assigned_driver_id=None,
                should_fallback_trigger_dispatch=True,
                fallback_reason="problem_incomplete",
            )

        try:
            res = self._assigner.assign_urgent(
                problem=problem,
                urgent_booking_ids=[booking_id],
                settings=settings,
            )
        except Exception as e:
            return DispatchNowResult(
                ok=False,
                error=str(e),
                should_fallback_trigger_dispatch=True,
                fallback_reason="assign_urgent_error",
            )

        assignments = getattr(res, "assignments", None)
        if not assignments:
            # Pas de chauffeur dispo: pas de fallback dans le code legacy
            return DispatchNowResult(ok=True, assigned_driver_id=None)

        try:
            apply_result = self._applier.apply_assignments(
                company_id=company_id,
                assignments=assignments,
                allow_reassign=True,
                respect_existing=False,
            )
        except Exception as e:
            return DispatchNowResult(
                ok=False,
                error=str(e),
                should_fallback_trigger_dispatch=True,
                fallback_reason="apply_error",
            )

        applied = apply_result.get("applied")
        first = applied[0] if isinstance(applied, list) and applied else None
        driver_id = first.get("driver_id") if isinstance(first, dict) else None
        try:
            driver_id_int = int(driver_id) if driver_id is not None else None
        except Exception:
            driver_id_int = None

        return DispatchNowResult(ok=True, assigned_driver_id=driver_id_int)
