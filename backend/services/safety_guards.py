"""Alias de compatibilité vers ``services.security.safety`` (patches de tests RL)."""

from __future__ import annotations

from services.security.safety import (
    SafetyGuards,
    SafetyThresholds,
    get_safety_guards,
    logger,
)

__all__ = ["SafetyGuards", "SafetyThresholds", "get_safety_guards", "logger"]
