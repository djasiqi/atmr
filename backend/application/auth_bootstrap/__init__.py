"""Bootstrap session (GET /auth/me) — read model et règles métier."""

# pyright: reportImportCycles=false, reportUnsupportedDunderAll=false
from __future__ import annotations

__all__ = ["GetBootstrapSessionUseCase"]


def __getattr__(name: str):
    """Import paresseux : évite de charger le use case tant qu'il n'est pas utilisé."""
    if name == "GetBootstrapSessionUseCase":
        from .get_bootstrap_session_use_case import GetBootstrapSessionUseCase as _UC

        return _UC
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
