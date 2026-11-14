"""Verrou S23 : modes semi_auto / fully_auto gelés (manuel uniquement).

Les flags restent OFF. Réveiller un mode = passer le flag à True.
"""

from __future__ import annotations

import os
from typing import Final

SEMI_AUTO_DISPATCH_ENABLED: Final[bool] = os.getenv(
    "SEMI_AUTO_DISPATCH_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}
FULLY_AUTO_DISPATCH_ENABLED: Final[bool] = os.getenv(
    "FULLY_AUTO_DISPATCH_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}


class DispatchModeLockedError(ValueError):
    """Tentative d'activer un mode de dispatch gelé."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        if mode == "semi_auto":
            msg = "Dispatch semi-automatique désactivé (verrou S23)."
        elif mode == "fully_auto":
            msg = "Dispatch automatique désactivé (verrou S23)."
        else:
            msg = f"Mode dispatch non autorisé: {mode}"
        super().__init__(msg)


def assert_dispatch_mode_switch_allowed(mode: str) -> None:
    """Abort si le mode demandé n'est pas autorisé.

    ``manual`` est toujours autorisé. Les autres modes exigent leur flag.
    """
    value = (mode or "").strip().lower()
    if value in {"", "manual"}:
        return
    if value in {"semi_auto", "semi-auto"}:
        if not SEMI_AUTO_DISPATCH_ENABLED:
            raise DispatchModeLockedError("semi_auto")
        return
    if value in {"fully_auto", "fully-auto", "auto"}:
        if not FULLY_AUTO_DISPATCH_ENABLED:
            raise DispatchModeLockedError("fully_auto")
        return
    raise DispatchModeLockedError(value)
