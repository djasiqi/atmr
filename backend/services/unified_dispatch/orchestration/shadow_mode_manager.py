"""Alias de compatibilité — ancien chemin orchestration/shadow_mode_manager."""

from __future__ import annotations

from services.unified_dispatch.shadow_mode.manager import ShadowModeManager
from services.unified_dispatch.shadow_mode.orchestrator import ShadowModeOrchestrator

__all__ = ["ShadowModeManager", "ShadowModeOrchestrator"]
