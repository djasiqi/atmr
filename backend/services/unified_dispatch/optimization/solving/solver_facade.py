"""Facade pour orchestrer heuristiques et solver."""

import logging
from typing import Any, cast

from services.unified_dispatch import heuristics, solver
from services.unified_dispatch.core import settings as ud_settings

logger = logging.getLogger(__name__)


class SolverFacade:
    """Facade pour orchestrer l'exécution des heuristiques et du solver."""

    def solve(
        self,
        problem: dict[str, Any],
        mode: str,
        settings: ud_settings.Settings,
        perf_collector: Any | None = None,
    ) -> tuple[list[Any], list[int], bool, bool]:
        """Exécute le pipeline de solving (heuristiques + solver + fallback).

        Args:
            problem: Dictionnaire contenant le problème
            mode: Mode de dispatch ("auto", "heuristic_only", "solver_only")
            settings: Paramètres de dispatch
            perf_collector: Collecteur de métriques de performance optionnel

        Returns:
            Tuple (assignments, unassigned_ids, used_heuristic, used_solver)
        """
        assignments: list[Any] = []
        unassigned_ids: list[int] = []
        used_heuristic = False
        used_solver = False

        # Helper pour étendre les assignments de manière unique
        def extend_unique(new_assignments: list[Any]) -> None:
            """Ajoute les nouvelles assignations en évitant les doublons."""
            existing_ids = {
                a.booking_id for a in assignments if hasattr(a, "booking_id")
            }
            for a in new_assignments:
                if hasattr(a, "booking_id") and a.booking_id not in existing_ids:
                    assignments.append(a)
                    existing_ids.add(a.booking_id)

        # Helper pour obtenir les IDs restants
        def remaining_ids_from(prob: dict[str, Any]) -> list[int]:
            """Retourne les IDs de bookings non assignés."""
            assigned_ids = {
                a.booking_id for a in assignments if hasattr(a, "booking_id")
            }
            bookings = prob.get("bookings", [])
            return [b.id for b in bookings if b.id not in assigned_ids]

        # 1) Heuristiques
        if mode in ("auto", "heuristic_only") and getattr(
            settings.features, "enable_heuristics", True
        ):
            try:
                if perf_collector:
                    with perf_collector.time_step("heuristic"):
                        h_res = heuristics.assign(problem, settings=settings)
                else:
                    h_res = heuristics.assign(problem, settings=settings)
                extend_unique(h_res.assignments)
                unassigned_ids = h_res.unassigned_booking_ids
                used_heuristic = True
                logger.info(
                    "[SolverFacade] Heuristic: %d assigned, %d unassigned",
                    len(h_res.assignments),
                    len(unassigned_ids),
                )
            except Exception as e:
                logger.warning("[SolverFacade] Heuristic failed: %s", e)

        # 2) Solver pour les restants
        rem = remaining_ids_from(problem)
        if (
            rem
            and mode in ("auto", "solver_only")
            and getattr(settings.features, "enable_solver", True)
        ):
            try:
                # Filtrer le problème pour ne garder que les restants
                filtered_problem = self._filter_problem(problem, rem, settings)

                # Warm-start: Injecter les assignments heuristiques si disponibles
                if assignments:
                    filtered_problem["heuristic_assignments"] = assignments

                if perf_collector:
                    with perf_collector.time_step("solver"):
                        s_res = solver.solve(filtered_problem, settings=settings)
                else:
                    s_res = solver.solve(filtered_problem, settings=settings)
                extend_unique(cast(list[Any], s_res.assignments))
                unassigned_ids = s_res.unassigned_booking_ids
                used_solver = True
                logger.info(
                    "[SolverFacade] Solver: +%d assigned, %d unassigned",
                    len(s_res.assignments),
                    len(unassigned_ids),
                )
            except Exception as e:
                logger.warning("[SolverFacade] Solver failed: %s", e)

        # 3) Fallback pour les restants
        rem = remaining_ids_from(problem)
        if rem:
            try:
                fb = heuristics.closest_feasible(problem, rem, settings=settings)
                extend_unique(fb.assignments)
                unassigned_ids = fb.unassigned_booking_ids
                logger.info(
                    "[SolverFacade] Fallback: +%d assigned, %d unassigned",
                    len(fb.assignments),
                    len(unassigned_ids),
                )
            except Exception as e:
                logger.warning("[SolverFacade] Fallback failed: %s", e)

        # Mettre à jour les IDs non assignés finaux
        unassigned_ids = remaining_ids_from(problem)

        return assignments, unassigned_ids, used_heuristic, used_solver

    def _filter_problem(
        self,
        problem: dict[str, Any],
        booking_ids: list[int],
        settings: ud_settings.Settings,  # noqa: ARG002 - Conservé pour compatibilité future
    ) -> dict[str, Any]:
        """Filtre le problème pour ne garder que les bookings spécifiés.

        Args:
            problem: Problème complet
            booking_ids: IDs des bookings à garder
            settings: Paramètres de dispatch (non utilisé actuellement, conservé pour compatibilité)

        Returns:
            Problème filtré
        """
        # Cette logique est complexe et dépend de l'implémentation dans engine.py
        # Pour l'instant, on délègue à la fonction existante si elle existe
        # Sinon, on fait un filtrage simple
        bookings = problem.get("bookings", [])
        filtered_bookings = [b for b in bookings if b.id in booking_ids]

        filtered_problem = problem.copy()
        filtered_problem["bookings"] = filtered_bookings

        return filtered_problem
