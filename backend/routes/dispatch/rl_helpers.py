"""Helpers partagés pour les routes RL (sans RLDispatchManager)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def rl_suggestion_generator_status() -> dict[str, Any]:
    """État du générateur de suggestions (RLSuggestionGenerator)."""
    try:
        from services.ml.rl.suggestion_generator import get_suggestion_generator

        gen = get_suggestion_generator()
        loaded = gen._is_model_loaded()
        return {
            "available": True,
            "loaded": loaded,
            "model_path": getattr(gen, "model_path", None),
            "message": None,
        }
    except Exception as e:
        logger.warning("[RL] suggestion generator unavailable: %s", e)
        return {
            "available": False,
            "loaded": False,
            "model_path": None,
            "message": str(e),
        }


def suggestions_observability_meta(
    generator: Any, duration_ms: float
) -> dict[str, Any]:
    """Métadonnées minimales (L09).

    Vocabulaire **stable** pour ``model_source`` (réponse JSON) :
    - ``dqn`` : fichier inférence chargé, chemin DQN actif.
    - ``basic_fallback`` : pas de modèle chargé ; suggestions issues de l'heuristique
      de secours dans ``RLSuggestionGenerator``.
    - Pour les réponses **cache**, le routeur fixe ``model_source`` à ``cache`` (pas ce helper).

    ``fallback_reason`` : ``null`` si ``dqn``, sinon ``model_missing`` (fichier absent ou load échoué).
    """
    loaded = bool(getattr(generator, "_is_model_loaded", lambda: False)())
    return {
        "duration_ms": round(duration_ms, 2),
        "model_source": "dqn" if loaded else "basic_fallback",
        "fallback_reason": None if loaded else "model_missing",
    }
