"""Types standardisés pour le système de dispatch.

Ce module définit les structures de données typées pour garantir
la cohérence et la type safety dans le système de dispatch.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DispatchResult:
    """Résultat standardisé d'un dispatch.

    Cette classe garantit que dispatch_run_id est toujours présent
    de manière cohérente dans tous les résultats de dispatch.

    Attributes:
        dispatch_run_id: ID du DispatchRun (peut être None si non créé)
        assignments: Liste des assignations créées
        unassigned: Liste des IDs de bookings non assignés
        bookings: Liste des bookings sérialisés
        drivers: Liste des drivers sérialisés
        meta: Métadonnées du dispatch (contient aussi dispatch_run_id)
        debug: Informations de debug (contient aussi dispatch_run_id)
        unassigned_reasons: Raisons détaillées de non-assignation (optionnel)
        performance_metrics: Métriques de performance (optionnel)
    """

    dispatch_run_id: int | None
    assignments: list[Any]
    unassigned: list[Any]
    bookings: list[Any]
    drivers: list[Any]
    meta: dict[str, Any]
    debug: dict[str, Any]
    unassigned_reasons: dict[int, list[str]] | None = None
    performance_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour compatibilité API.

        Garantit que dispatch_run_id est toujours présent dans meta et debug.

        Returns:
            Dictionnaire avec la structure standardisée du résultat
        """
        result: dict[str, Any] = {
            "dispatch_run_id": self.dispatch_run_id,
            "assignments": self.assignments,
            "unassigned": self.unassigned,
            "bookings": self.bookings,
            "drivers": self.drivers,
            "meta": {
                **self.meta,
                "dispatch_run_id": self.dispatch_run_id,  # ✅ Toujours dans meta
            },
            "debug": {
                **self.debug,
                "dispatch_run_id": self.dispatch_run_id,  # ✅ Toujours dans debug
            },
        }

        # Ajouter les champs optionnels s'ils sont présents
        if self.unassigned_reasons is not None:
            result["unassigned_reasons"] = self.unassigned_reasons

        if self.performance_metrics is not None:
            result["performance_metrics"] = self.performance_metrics

        return result
