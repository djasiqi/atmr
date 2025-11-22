# backend/services/unified_dispatch/score_fusion.py
"""Score fusion heuristique + RL."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


def fuse_scores(
    heuristic_score: float, rl_score: float, alpha: float
) -> Tuple[float, Dict[str, Any]]:
    """Fusionne les scores heuristique et RL.

    Args:
        heuristic_score: Score heuristique (0-100)
        rl_score: Score RL (0-1)
        alpha: Poids RL (0 = heuristique pure, 1 = RL pur)

    Returns:
        Tuple (final_score, breakdown)
    """
    # Normaliser RL score de 0-1 vers 0-100
    rl_score_normalized = rl_score * 100

    # Fusion linéaire
    final_score = (1 - alpha) * heuristic_score + alpha * rl_score_normalized

    breakdown = {
        "heuristic": heuristic_score,
        "rl": rl_score_normalized,
        "alpha": alpha,
        "final": final_score,
    }

    logger.debug(
        "[ScoreFusion] heur=%.2f, rl=%.2f, alpha=%.2f → final=%.2f",
        heuristic_score,
        rl_score_normalized,
        alpha,
        final_score,
    )

    return final_score, breakdown


def should_use_rl_score(
    enable_rl: bool, enable_rl_apply: bool, has_rl_score: bool
) -> bool:
    """Détermine si le score RL doit être utilisé.

    Args:
        enable_rl: Feature flag pour activer RL
        enable_rl_apply: Feature flag pour appliquer RL (pas seulement shadow)
        has_rl_score: Indique si un score RL est disponible

    Returns:
        True si le score RL doit être utilisé
    """
    return enable_rl and enable_rl_apply and has_rl_score
