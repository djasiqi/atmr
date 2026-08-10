"""Alias de compatibilité — ancien ``autonomous_manager.py`` → ``utils/autonomous``."""

from __future__ import annotations

from services.unified_dispatch.utils.autonomous import (
    AutonomousDispatchManager,
    get_manager_for_company,
)
from services.unified_dispatch.utils.suggestions import apply_suggestion

__all__ = ["AutonomousDispatchManager", "apply_suggestion", "get_manager_for_company"]
