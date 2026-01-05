# backend/services/unified_dispatch/orchestration/assignment_applier_wrapper.py
"""Wrapper pour l'application des assignations.

Ce module fournit un wrapper autour de AssignmentApplier pour l'application
des assignations en base de données. Il est responsable de :
- L'application des assignations finales en DB
- L'émission d'événements Socket.IO pour le temps réel
- La mesure des performances de persistence

Side-effects:
    - Accès DB (écriture Assignment)
    - Socket.IO: Émissions d'événements temps réel
    - Métriques: Performance persistence via perf_collector
"""

from __future__ import annotations

import logging
from typing import Any

from services.unified_dispatch.assignment.assignment_applier import AssignmentApplier

logger = logging.getLogger(__name__)


class AssignmentApplierWrapper:
    """Wrapper pour l'application des assignations.

    Cette classe fournit une interface simplifiée pour appliquer les
    assignations finales en base de données et émettre les événements
    temps réel correspondants.

    Exemple:
        >>> wrapper = AssignmentApplierWrapper()
        >>> wrapper.apply(
        ...     company=company,
        ...     final_assignments=assignments,
        ...     dispatch_run_id=42,
        ...     perf_collector=perf_collector
        ... )
    """

    def apply(
        self,
        company: Any,
        final_assignments: list[Any],
        dispatch_run_id: int | None,
        perf_collector: Any | None,
    ) -> None:
        """Applique les assignations en base de données.

        Applique les assignations finales en base de données via
        AssignmentApplier et émet les événements Socket.IO correspondants.
        Mesure également le temps de persistence si un perf_collector
        est fourni.

        Args:
            company: Objet Company (doit être non-None)
            final_assignments: List des assignations finales à appliquer
            dispatch_run_id: ID du DispatchRun (peut être None)
            perf_collector: Collecteur de métriques de performance (optionnel)

        Side-effects:
            - Accès DB (écriture Assignment)
            - Socket.IO: Émissions d'événements temps réel
            - Métriques: Performance persistence (si perf_collector fourni)
        """
        if not company:
            return

        assignment_applier = AssignmentApplier()

        # Mesurer le temps de persistence
        if perf_collector:
            with perf_collector.time_step("persistence"):
                assignment_applier.apply_and_emit(
                    company, final_assignments, dispatch_run_id
                )
        else:
            assignment_applier.apply_and_emit(
                company, final_assignments, dispatch_run_id
            )
