"""Replayer de journée pour simulations offline.

Permet de rejouer une journée N fois avec stochastic traffic
et comparer différentes politiques de dispatch.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Constantes
STOCHASTIC_NOISE_PERCENT = 0.15  # Bruit aléatoire ±15% pour trafic stochastique


@dataclass
class SimulationResult:
    """Résultat d'une simulation."""

    policy_name: str
    total_bookings: int
    assignments_count: int
    unassigned_count: int
    total_distance_km: float
    fairness_score: float
    avg_on_time_rate: float
    execution_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "total_bookings": self.total_bookings,
            "assignments_count": self.assignments_count,
            "unassigned_count": self.unassigned_count,
            "total_distance_km": round(self.total_distance_km, 2),
            "fairness_score": round(self.fairness_score, 3),
            "avg_on_time_rate": round(self.avg_on_time_rate, 3),
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


class DayReplayer:
    """Replayer de journée avec trafic stochastique.

    Permet de:
    - Rejouer une journée N fois avec variabilité
    - Comparer différentes politiques (heuristique, RL, hybrid)
    - Tester nouvelles politiques avant déploiement
    """

    def __init__(self, stochastic_traffic: bool = True):
        super().__init__()
        """Initialise le replayer.

        Args:
            stochastic_traffic: Activer bruit stochastique sur temps trajet
        """
        self.stochastic_traffic = stochastic_traffic
        self.results: List[SimulationResult] = []

    def replay_day(
        self,
        target_date: str,
        bookings: List[Any],
        drivers: List[Any],
        n_iterations: int = 3,
        policies: List[str] | None = None,
    ) -> List[SimulationResult]:
        """Rejoue une journée N fois avec différentes politiques.

        Args:
            target_date: Date à rejouer
            bookings: Liste des bookings
            drivers: Liste des drivers
            n_iterations: Nombre d'itérations
            policies: Politiques à tester (par défaut: ["heuristic", "solver", "rl"])

        Returns:
            Résultats par politique
        """
        if policies is None:
            policies = ["heuristic", "solver", "rl"]

        logger.info(
            "[DayReplayer] Replaying day %s: %d bookings, %d drivers, %d policies, %d iterations",
            target_date,
            len(bookings),
            len(drivers),
            len(policies),
            n_iterations,
        )

        results = []

        for policy in policies:
            policy_results = []

            for iteration in range(n_iterations):
                logger.debug(
                    "[DayReplayer] Policy %s, iteration %d/%d",
                    policy,
                    iteration + 1,
                    n_iterations,
                )

                # Créer problème avec variabilité stochastique
                problem = self._create_problem_with_noise(bookings, drivers, iteration)

                # Exécuter politique
                result = self._run_policy(policy, problem)

                policy_results.append(result)

            # Agréger résultats (moyenne)
            aggregated = self._aggregate_results(policy_results, policy)
            results.append(aggregated)

            logger.info(
                "[DayReplayer] Policy %s: avg_bookings=%d, avg_fairness=%.2f",
                policy,
                aggregated.assignments_count,
                aggregated.fairness_score,
            )

        self.results = results
        return results

    def _create_problem_with_noise(
        self, bookings: List[Any], drivers: List[Any], iteration: int
    ) -> Dict[str, Any]:
        """Crée un problème avec bruit stochastique.

        Args:
            bookings: Listes des bookings
            drivers: Liste des drivers
            iteration: Numéro d'itération pour reproductibilité

        Returns:
            Problème avec time_matrix bruitée
        """
        # Seed pour reproductibilité
        random.seed(iteration * 42)
        np.random.seed(iteration * 42)

        # Créer time_matrix de base
        n_bookings = len(bookings)

        # Matrice base (simplifiée: utiliser distances Haversine)
        time_matrix = []

        for _ in range(n_bookings + 1):  # +1 pour depot
            row = []
            for _ in range(n_bookings + 1):
                base_time = 60  # 60 min par défaut

                # Ajouter bruit stochastique si activé
                if self.stochastic_traffic:
                    noise = np.random.normal(1.0, STOCHASTIC_NOISE_PERCENT)
                    noisy_time = int(base_time * noise)
                else:
                    noisy_time = base_time

                row.append(noisy_time)
            time_matrix.append(row)

        return {
            "bookings": bookings,
            "drivers": drivers,
            "time_matrix": time_matrix,
            "iteration": iteration,
        }

    def _run_policy(self, policy: str, problem: Dict[str, Any]) -> SimulationResult:
        """Exécute une politique sur un problème.

        Args:
            policy: Nom de la politique ("heuristic", "solver", "rl")
            problem: Problème à résoudre

        Returns:
            SimulationResult
        """
        import time

        start_time = time.time()

        bookings = problem["bookings"]

        # Selon politique, appeler le bon resolver
        if policy == "heuristic":
            assignments = self._run_heuristic(problem)
        elif policy == "solver":
            assignments = self._run_solver(problem)
        elif policy == "rl":
            assignments = self._run_rl(problem)
        else:
            logger.warning("[DayReplayer] Policy inconnue: %s", policy)
            assignments = []

        execution_time_ms = (time.time() - start_time) * 1000

        # Calculer métriques
        fairness = calculate_fairness_score(assignments)
        efficiency = calculate_efficiency_score(assignments)

        return SimulationResult(
            policy_name=policy,
            total_bookings=len(bookings),
            assignments_count=len(assignments),
            unassigned_count=len(bookings) - len(assignments),
            total_distance_km=float(efficiency * 10),  # Approximation
            fairness_score=float(fairness),
            avg_on_time_rate=0.85,  # TODO: calculer depuis historique
            execution_time_ms=float(execution_time_ms),
        )

    def _run_heuristic(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Exécute politique heuristique."""
        try:
            from services.unified_dispatch.heuristics import assign
            from services.unified_dispatch.settings import Settings

            settings = Settings()
            result = assign(problem, settings)

            # Convertir HeuristicAssignment en dict
            assignments = result.assignments if result else []
            return [
                {"driver_id": a.driver_id, "booking_id": a.booking_id}
                for a in assignments
            ]
        except Exception as e:
            logger.error("[DayReplayer] Erreur heuristique: %s", e)
            return []

    def _run_solver(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Exécute politique solver OR-Tools."""
        try:
            from services.unified_dispatch.settings import Settings
            from services.unified_dispatch.solver import solve

            settings = Settings()
            result = solve(problem, settings)

            # Convertir SolverAssignment en dict
            assignments = result.assignments if result else []
            return [
                {"driver_id": a.driver_id, "booking_id": a.booking_id}
                for a in assignments
            ]
        except Exception as e:
            logger.error("[DayReplayer] Erreur solver: %s", e)
            return []

    def _run_rl(self, problem: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Exécute politique RL."""
        # TODO: Intégrer avec RLSettings
        logger.warning("[DayReplayer] RL policy non implémentée, fallback heuristique")
        return self._run_heuristic(problem)

    def _aggregate_results(
        self, results: List[SimulationResult], policy_name: str
    ) -> SimulationResult:
        """Agrège plusieurs résultats en un seul."""
        avg_assignments = np.mean([r.assignments_count for r in results])
        avg_unassigned = np.mean([r.unassigned_count for r in results])
        avg_distance = np.mean([r.total_distance_km for r in results])
        avg_fairness = np.mean([r.fairness_score for r in results])
        avg_on_time = np.mean([r.avg_on_time_rate for r in results])
        avg_time = np.mean([r.execution_time_ms for r in results])

        # Utiliser un résultat représentatif
        representative = results[0]

        return SimulationResult(
            policy_name=policy_name,
            total_bookings=representative.total_bookings,
            assignments_count=int(avg_assignments),
            unassigned_count=int(avg_unassigned),
            total_distance_km=float(avg_distance),
            fairness_score=float(avg_fairness),
            avg_on_time_rate=float(avg_on_time),
            execution_time_ms=float(avg_time),
        )


# Import helper functions depuis pareto_front
try:
    from services.unified_dispatch.pareto_front import (
        calculate_efficiency_score,
        calculate_fairness_score,
    )
except ImportError:
    # Fallback si imports échouent
    def calculate_efficiency_score(
        assignments: List[Dict[str, Any]], _problem: Dict[str, Any] | None = None
    ) -> float:
        return 1.0 / (1.0 + len(assignments))

    def calculate_fairness_score(assignments: List[Dict[str, Any]]) -> float:  # noqa: ARG001
        return 0.5  # Placeholder
