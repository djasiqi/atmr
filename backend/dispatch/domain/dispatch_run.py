"""Agrégat racine : DispatchRun."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from dispatch.domain.dispatch_run_id import DispatchRunId
from dispatch.domain.value_objects import DispatchMetrics, DispatchStatus


@dataclass
class DispatchRun:
    """Agrégat racine : Exécution d'un dispatch.

    Responsabilités :
    - Gérer l'exécution d'un dispatch pour une entreprise et un jour
    - Gérer les assignations générées
    - Calculer les métriques
    - Appliquer les invariants métier
    """

    id: DispatchRunId
    company_id: int
    day: date
    status: DispatchStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None
    config: dict[str, Any] | None = None
    metrics: DispatchMetrics | None = None

    def start(self) -> None:
        """Démarre le dispatch (méthode métier)."""
        if not self.status.can_start():
            raise ValueError(f"Cannot start dispatch in status {self.status.value}")
        self.status = DispatchStatus(value="RUNNING")
        self.started_at = datetime.now()

    def complete(self, metrics: DispatchMetrics | None = None) -> None:
        """Complète le dispatch avec métriques (méthode métier)."""
        if not self.status.can_complete():
            raise ValueError(f"Cannot complete dispatch in status {self.status.value}")
        self.status = DispatchStatus(value="COMPLETED")
        self.completed_at = datetime.now()
        if metrics:
            self.metrics = metrics

    def fail(self, reason: str | None = None) -> None:
        """Marque le dispatch comme échoué (méthode métier)."""
        if self.status.is_final():
            raise ValueError(
                f"Cannot fail dispatch in final status {self.status.value}"
            )
        self.status = DispatchStatus(value="FAILED")
        self.completed_at = datetime.now()
        # Optionnel : stocker la raison dans config
        if reason and self.config is None:
            self.config = {"error": reason}
        elif reason:
            self.config = {**(self.config or {}), "error": reason}

    def validate(self) -> bool:
        """Valide les invariants métier."""
        # Invariant 1: Un DispatchRun est unique par (company_id, day)
        # (géré par contrainte DB, mais on peut vérifier ici si nécessaire)

        # Invariant 2: Si completed_at est présent, status doit être COMPLETED ou FAILED
        if self.completed_at is not None and not self.status.is_final():
            return False

        # Invariant 3: Si started_at est présent, status doit être RUNNING, COMPLETED ou FAILED
        if self.started_at is not None and self.status.is_pending():
            return False

        # Invariant 4: company_id doit être positif
        return self.company_id > 0
