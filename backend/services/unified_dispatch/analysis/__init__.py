"""Alias de compatibilité — ancien package ``analysis`` → ``validation/analysis``."""

from __future__ import annotations

import services.unified_dispatch.validation.analysis as _validation_analysis

# Délègue la résolution des sous-modules à validation.analysis
__path__ = list(_validation_analysis.__path__)  # type: ignore[misc, name-defined]

from services.unified_dispatch.validation.analysis import UnassignedAnalyzer

__all__ = ["UnassignedAnalyzer"]
