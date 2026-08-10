"""Alias de compatibilité vers ``services.unified_dispatch.ml.predictor``.

Réexporte l'intégralité du module pour que les imports et patches de tests
sur l'ancien chemin ``ml_predictor`` restent valides.
"""

from __future__ import annotations

from services.unified_dispatch.ml.predictor import *  # noqa: F401,F403
