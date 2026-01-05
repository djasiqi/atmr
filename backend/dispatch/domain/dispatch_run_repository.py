"""Interface du repository pour DispatchRun (port)."""

from __future__ import annotations

from datetime import date
from typing import Protocol

from dispatch.domain.dispatch_run import DispatchRun
from dispatch.domain.dispatch_run_id import DispatchRunId


class DispatchRunRepository(Protocol):
    """Port (interface) pour le repository de DispatchRun.

    L'implémentation sera dans infrastructure/repositories/.
    """

    def save(self, dispatch_run: DispatchRun) -> None:
        """Sauvegarde un dispatch run."""
        ...

    def find_by_id(self, run_id: DispatchRunId) -> DispatchRun | None:
        """Trouve un dispatch run par ID."""
        ...

    def find_by_company_and_day(self, company_id: int, day: date) -> DispatchRun | None:
        """Trouve un dispatch run par entreprise et jour."""
        ...
