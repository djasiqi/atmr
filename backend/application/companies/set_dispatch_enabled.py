from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class _CompanyLike(Protocol):
    id: int | None
    dispatch_enabled: bool
    dispatch_mode: Any


@dataclass(frozen=True, slots=True)
class SetDispatchEnabledResult:
    ok: bool
    error: dict[str, str] | None = None
    status_code: int | None = None
    enabled: bool | None = None
    company_id: int | None = None
    should_trigger_dispatch: bool = False
    trigger_reason: str | None = None


class SetDispatchEnabledUseCase:
    """Use-case Application: activer/désactiver le dispatch automatique."""

    def execute(
        self, company: _CompanyLike, *, enabled: bool, reason: str
    ) -> SetDispatchEnabledResult:
        if not hasattr(company, "dispatch_enabled"):
            return SetDispatchEnabledResult(
                ok=False,
                error={"error": "Le champ 'dispatch_enabled' n'existe pas sur Company"},
                status_code=400,
            )

        # Invariant métier : MANUAL ⇒ dispatch_enabled=false.
        # On refuse d'activer le dispatch tant que la société est en mode MANUAL.
        if bool(enabled) and self._is_manual_mode(company):
            return SetDispatchEnabledResult(
                ok=False,
                error={
                    "error": (
                        "Impossible d'activer le dispatch en mode MANUAL. "
                        "Changez d'abord le mode de dispatch (SEMI_AUTO ou "
                        "FULLY_AUTO)."
                    )
                },
                status_code=409,
            )

        company.dispatch_enabled = bool(enabled)

        cid_obj: Any = getattr(company, "id", None)
        try:
            cid = int(cid_obj) if cid_obj is not None else None
        except Exception:
            cid = None

        should_trigger = bool(enabled) and cid is not None
        return SetDispatchEnabledResult(
            ok=True,
            enabled=bool(getattr(company, "dispatch_enabled", False)),
            company_id=cid,
            should_trigger_dispatch=should_trigger,
            trigger_reason=reason if should_trigger else None,
        )

    @staticmethod
    def _is_manual_mode(company: _CompanyLike) -> bool:
        """True si la société est en mode dispatch MANUAL.

        Tolérant aux représentations possibles (enum DispatchMode ou chaîne).
        """
        mode = getattr(company, "dispatch_mode", None)
        if mode is None:
            return False
        value = getattr(mode, "value", mode)
        return str(value).strip().lower() == "manual"
