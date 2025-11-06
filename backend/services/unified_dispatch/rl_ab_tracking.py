"""✅ B4: Tracking A/B RL pour rollout progressif.

Objectif: Mesurer les gains RL vs heuristique pour décision rollout.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, List

logger = logging.getLogger(__name__)

# Seuil de gain minimum pour augmenter le rollout RL (en points de qualité)
MIN_RL_GAIN_FOR_ROLLOUT = 3.0


@dataclass
class RLABResult:
    """Résultat A/B test pour une entreprise."""
    
    company_id: int
    date: datetime
    quality_score_heuristic: float
    quality_score_rl: float
    delta_quality: float  # RL - Heuristic
    assignment_rate_heuristic: float
    assignment_rate_rl: float
    delta_assignment: float
    pooling_rate_heuristic: float
    pooling_rate_rl: float
    delta_pooling: float
    gain_detected: bool  # True si delta_quality >= MIN_RL_GAIN_FOR_ROLLOUT


@dataclass
class RLGainTracker:
    """Tracker pour gains RL historiques."""
    
    company_id: int
    gains_history: List[RLABResult]
    median_gain: float = 0.0
    should_increase_rollout: bool = False
    
    def record_result(self, result: RLABResult) -> None:
        """Enregistre un résultat A/B."""
        self.gains_history.append(result)
        
        # Recalculer median gain (14 derniers jours)
        recent_gains = [
            r.delta_quality for r in self.gains_history[-14:]
            if r.gain_detected
        ]
        
        if recent_gains:
            sorted_gains = sorted(recent_gains)
            n = len(sorted_gains)
            self.median_gain = sorted_gains[n // 2] if n % 2 == 1 else (sorted_gains[n // 2 - 1] + sorted_gains[n // 2]) / 2
            
            # ✅ B4: Augmenter rollout si gain médian >= seuil minimum
            self.should_increase_rollout = self.median_gain >= MIN_RL_GAIN_FOR_ROLLOUT
            
            logger.info(
                "[B4] Company %d: median_gain=%.2f, should_increase_rollout=%s",
                self.company_id, self.median_gain, self.should_increase_rollout
            )


def get_or_create_rl_tracker(company_id: int) -> RLGainTracker:
    """Récupère ou crée un tracker RL pour une entreprise."""
    # TODO: Persister en DB si nécessaire
    return RLGainTracker(company_id=company_id, gains_history=[])


def evaluate_rl_gain(company_id: int, heuristic_result: Any, rl_result: Any) -> RLABResult:
    """Évalue le gain RL vs heuristique.
    
    Args:
        company_id: ID de l'entreprise
        heuristic_result: Résultat avec heuristique uniquement
        rl_result: Résultat avec RL appliqué
        
    Returns:
        RLABResult avec delta
    """
    quality_heuristic = heuristic_result.get("quality_score", 0.0)
    quality_rl = rl_result.get("quality_score", 0.0)
    delta_quality = quality_rl - quality_heuristic
    
    assignment_heuristic = heuristic_result.get("assignment_rate", 0.0)
    assignment_rl = rl_result.get("assignment_rate", 0.0)
    delta_assignment = assignment_rl - assignment_heuristic
    
    pooling_heuristic = heuristic_result.get("pooling_rate", 0.0)
    pooling_rl = rl_result.get("pooling_rate", 0.0)
    delta_pooling = pooling_rl - pooling_heuristic
    
    return RLABResult(
        company_id=company_id,
        date=datetime.now(UTC),
        quality_score_heuristic=quality_heuristic,
        quality_score_rl=quality_rl,
        delta_quality=delta_quality,
        assignment_rate_heuristic=assignment_heuristic,
        assignment_rate_rl=assignment_rl,
        delta_assignment=delta_assignment,
        pooling_rate_heuristic=pooling_heuristic,
        pooling_rate_rl=pooling_rl,
        delta_pooling=delta_pooling,
        gain_detected=delta_quality >= MIN_RL_GAIN_FOR_ROLLOUT
    )

