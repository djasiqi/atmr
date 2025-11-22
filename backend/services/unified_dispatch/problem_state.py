"""Gestion centralisée de l'état des chauffeurs pendant le dispatch.

Élimine la duplication de code dans heuristics.py, solver.py et data.py
en centralisant la logique de gestion de l'état des chauffeurs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ProblemState:
    """Encapsule l'état dynamique des chauffeurs pendant un run de dispatch.

    Attributs:
        busy_until: Timestamp (minutes) jusqu'à quand chaque chauffeur est occupé
        scheduled_times: Liste des horaires (minutes) déjà assignés à chaque chauffeur
        proposed_load: Nombre de courses proposées à chaque chauffeur dans ce run
    """

    busy_until: Dict[int, int] = field(default_factory=dict)
    scheduled_times: Dict[int, List[int]] = field(default_factory=dict)
    proposed_load: Dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_problem(cls, problem: Dict[str, Any], drivers: List[Any]) -> ProblemState:
        """Crée un ProblemState à partir d'un dict problem et d'une liste de drivers.

        Args:
            problem: Dict avec les clés optionnelles busy_until, driver_scheduled_times, proposed_load
            drivers: Liste des chauffeurs disponibles

        Returns:
            ProblemState initialisé avec les valeurs du problem ou des defaults

        """
        # Récupérer les états précédents ou initialiser vides
        previous_busy = problem.get("busy_until", {})
        previous_times = problem.get("driver_scheduled_times", {})
        previous_load = problem.get("proposed_load", {})

        # Initialiser les dicts avec tous les drivers
        driver_ids = [int(d.id) for d in drivers]

        state = cls(
            busy_until={did: previous_busy.get(did, 0) for did in driver_ids},
            scheduled_times={
                did: list(previous_times.get(did, [])) for did in driver_ids
            },
            proposed_load={did: previous_load.get(did, 0) for did in driver_ids},
        )

        if previous_busy or previous_times or previous_load:
            logger.debug(
                "[ProblemState] État récupéré: %d busy_until, %d scheduled_times, %d proposed_load",
                len(previous_busy),
                len(previous_times),
                len(previous_load),
            )

        return state

    def to_dict(self) -> Dict[str, Any]:
        """Convertit le ProblemState en dict pour le stocker dans problem.

        Returns:
            Dict avec les clés busy_until, driver_scheduled_times, proposed_load

        """
        return {
            "busy_until": dict(self.busy_until),
            "driver_scheduled_times": {
                k: list(v) for k, v in self.scheduled_times.items()
            },
            "proposed_load": dict(self.proposed_load),
        }

    def update_problem(self, problem: Dict[str, Any]) -> None:
        """Met à jour le dict problem avec l'état actuel.

        Args:
            problem: Dict à mettre à jour (modifié in-place)

        """
        problem.update(self.to_dict())

    def is_driver_busy_at(self, driver_id: int, time_min: int) -> bool:
        """Vérifie si un chauffeur est occupé à un moment donné.

        Args:
            driver_id: ID du chauffeur
            time_min: Timestamp en minutes

        Returns:
            True si le chauffeur est occupé, False sinon

        """
        return time_min < self.busy_until.get(driver_id, 0)

    def has_time_conflict(
        self, driver_id: int, time_min: int, min_gap_minutes: int = 30
    ) -> bool:
        """Vérifie si l'ajout d'une course à time_min crée un conflit avec les courses existantes.

        Args:
            driver_id: ID du chauffeur
            time_min: Timestamp de la nouvelle course (minutes)
            min_gap_minutes: Écart minimum requis entre deux courses

        Returns:
            True si conflit détecté, False sinon

        """
        existing_times = self.scheduled_times.get(driver_id, [])

        for existing_time in existing_times:
            if abs(time_min - existing_time) < min_gap_minutes:
                logger.debug(
                    "[ProblemState] Conflit détecté pour driver #%d: course à %dmin vs nouvelle à %dmin (écart: %dmin < %dmin)",
                    driver_id,
                    existing_time,
                    time_min,
                    abs(time_min - existing_time),
                    min_gap_minutes,
                )
                return True

        return False

    def assign_booking(
        self, driver_id: int, start_time_min: int, end_time_min: int
    ) -> None:
        """Enregistre l'assignation d'une course à un chauffeur.

        Met à jour :
        - busy_until: jusqu'à end_time_min
        - scheduled_times: ajoute start_time_min
        - proposed_load: +1

        Args:
            driver_id: ID du chauffeur
            start_time_min: Début de la course (minutes)
            end_time_min: Fin de la course (minutes)

        """
        # Mettre à jour busy_until
        self.busy_until[driver_id] = max(
            self.busy_until.get(driver_id, 0), end_time_min
        )

        # Ajouter le scheduled_time
        if driver_id not in self.scheduled_times:
            self.scheduled_times[driver_id] = []
        self.scheduled_times[driver_id].append(start_time_min)

        # Incrémenter le load
        self.proposed_load[driver_id] = self.proposed_load.get(driver_id, 0) + 1

        logger.debug(
            "[ProblemState] Driver #%d assigné: start=%d, end=%d, busy_until=%d, load=%d",
            driver_id,
            start_time_min,
            end_time_min,
            self.busy_until[driver_id],
            self.proposed_load[driver_id],
        )

    def get_driver_load(
        self, driver_id: int, fairness_counts: Dict[int, int] | None = None
    ) -> int:
        """Retourne la charge totale d'un chauffeur (proposée + déjà assignée).

        Args:
            driver_id: ID du chauffeur
            fairness_counts: Dict optionnel avec le nombre de courses déjà assignées aujourd'hui

        Returns:
            Charge totale du chauffeur

        """
        proposed = self.proposed_load.get(driver_id, 0)
        existing = fairness_counts.get(driver_id, 0) if fairness_counts else 0
        return proposed + existing

    def can_assign(
        self,
        driver_id: int,
        start_time_min: int,
        max_bookings_per_driver: int,
        fairness_counts: Dict[int, int] | None = None,
        min_gap_minutes: int = 30,
    ) -> tuple[bool, str | None]:
        """Vérifie si on peut assigner une nouvelle course à un chauffeur.

        Args:
            driver_id: ID du chauffeur
            start_time_min: Heure de début de la course (minutes)
            max_bookings_per_driver: Nombre maximum de courses par chauffeur
            fairness_counts: Courses déjà assignées aujourd'hui
            min_gap_minutes: Écart minimum entre deux courses

        Returns:
            tuple (can_assign: bool, reason: Optional[str])
            - (True, None) si assignation possible
            - (False, "raison") sinon

        """
        # Vérifier le plafond de courses
        total_load = self.get_driver_load(driver_id, fairness_counts)
        if total_load >= max_bookings_per_driver:
            return (
                False,
                f"max_capacity_reached ({total_load}/{max_bookings_per_driver})",
            )

        # Vérifier si le chauffeur est déjà occupé
        if self.is_driver_busy_at(driver_id, start_time_min):
            busy_until = self.busy_until.get(driver_id, 0)
            return False, f"driver_busy_until ({busy_until}min)"

        # Vérifier les conflits d'horaires
        if self.has_time_conflict(driver_id, start_time_min, min_gap_minutes):
            return False, "time_conflict"

        return True, None

    def reset_driver(self, driver_id: int) -> None:
        """Réinitialise l'état d'un chauffeur (utile pour les tests ou rollback).

        Args:
            driver_id: ID du chauffeur à réinitialiser

        """
        self.busy_until[driver_id] = 0
        self.scheduled_times[driver_id] = []
        self.proposed_load[driver_id] = 0

        logger.debug("[ProblemState] Driver #%s réinitialisé", driver_id)

    def get_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'état actuel (utile pour le debug).

        Returns:
            Dict avec les statistiques de l'état

        """
        total_assignments = sum(self.proposed_load.values())
        active_drivers = len([v for v in self.proposed_load.values() if v > 0])

        return {
            "total_assignments": total_assignments,
            "active_drivers": active_drivers,
            "total_drivers": len(self.proposed_load),
            "avg_load": total_assignments / max(active_drivers, 1),
            "max_load": max(self.proposed_load.values(), default=0),
            "busy_drivers": len([v for v in self.busy_until.values() if v > 0]),
        }

    def __repr__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        """Représentation string pour le debug."""
        summary = self.get_summary()
        return (
            f"ProblemState("
            f"assignments={summary['total_assignments']}, "
            f"active_drivers={summary['active_drivers']}/{summary['total_drivers']}, "
            f"avg_load={summary['avg_load']:.1f})"
        )


# Fonctions helper pour la compatibilité avec le code existant
def extract_state_from_problem(
    problem: Dict[str, Any], drivers: List[Any]
) -> ProblemState:
    """Fonction helper pour extraire un ProblemState depuis un dict problem.
    Alias de ProblemState.from_problem() pour compatibilité.
    """
    return ProblemState.from_problem(problem, drivers)


def inject_state_into_problem(state: ProblemState, problem: Dict[str, Any]) -> None:
    """Fonction helper pour injecter un ProblemState dans un dict problem.
    Alias de state.update_problem() pour compatibilité.
    """
    state.update_problem(problem)
