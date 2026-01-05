from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class _DriverLike(Protocol):
    id: int | None


@dataclass(frozen=True, slots=True)
class DeleteCompanyDriverResult:
    ok: bool
    should_trigger_dispatch: bool = False


class DeleteCompanyDriverUseCase:
    """Use-case Application: suppression d'un chauffeur (la route gère db.session.delete)."""

    def execute(self, driver: _DriverLike) -> DeleteCompanyDriverResult:
        _ = driver
        return DeleteCompanyDriverResult(ok=True, should_trigger_dispatch=True)
