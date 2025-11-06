# backend/services/unified_dispatch/ab_router.py
"""AB Router pour rollout progressif RL."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ABRouter:
    """Routage A/B pour rollout progressif RL."""
    
    def __init__(self, settings: Any):
        """Initialise le router A/B.
        
        Args:
            settings: Configuration settings avec RLSettings
        """
        super().__init__()
        self.settings = settings
        
        # ✅ B4: Rollout progressif: shadow → 10% → 50%
        # Vérifier variable d'environnement pour rollout
        import os
        rollout_pct = int(os.getenv("RL_ROLLOUT_PERCENTAGE", "10"))  # 10% par défaut
        self.bucket_size = rollout_pct
    
    def should_apply_rl(self, company_id: int) -> bool:
        """Détermine si RL doit être appliqué pour cette entreprise.
        
        Args:
            company_id: ID de l'entreprise
            
        Returns:
            True si RL doit être appliqué
        """
        # ✅ B4: Routing déterministe par company_id (buckets)
        # Modulo 100 pour buckets déterministes (0-99)
        bucket = company_id % 100
        
        # Activer pour N% des companies
        should_apply = bucket < self.bucket_size
        
        logger.debug(
            "[ABRouter] company=%d, bucket=%d/%d, should_apply_rl=%s (rollout=%d%%)",
            company_id, bucket, 100, should_apply, self.bucket_size
        )
        
        return should_apply
    
    def get_rl_settings(self, company_id: int) -> Optional[Dict[str, Any]]:
        """Retourne les paramètres RL pour une entreprise.
        
        Args:
            company_id: ID de l'entreprise
            
        Returns:
            Dict des paramètres RL ou None si pas d'activation
        """
        if not self.should_apply_rl(company_id):
            return None
        
        rl_config = getattr(self.settings, "rl", None)
        alpha = getattr(rl_config, "alpha", 0.2) if rl_config else 0.2
        
        return {
            "enable_rl_apply": True,
            "alpha": alpha,
            "company_id": company_id
        }
    
    def should_apply_with_quality_check(
        self,
        company_id: int,
        projected_quality_score: float
    ) -> bool:
        """Vérifie si RL peut être appliqué en tenant compte du quality_score prévisionnel.
        
        Args:
            company_id: ID de l'entreprise
            projected_quality_score: Quality score prévisionnel
            
        Returns:
            True si RL peut être appliqué
        """
        # Vérifier le bucket
        if not self.should_apply_rl(company_id):
            return False
        
        # Vérifier le quality_score prévisionnel
        rl_config = getattr(self.settings, "rl", None)
        min_quality_score = (
            getattr(rl_config, "min_quality_score", 70.0) 
            if rl_config 
            else 70.0
        )
        
        if projected_quality_score < min_quality_score:
            logger.warning(
                "[ABRouter] Company %d: projected quality_score (%.1f) < threshold (%.1f), skipping RL apply",
                company_id, projected_quality_score, min_quality_score
            )
            return False
        
        return True
    
    def should_apply_with_safety_guards(
        self,
        company_id: int,
        projected_quality_score: float,
        min_pickup_minutes: float | None = None
    ) -> bool:
        """✅ B4: Vérifie avec tous les garde-fous (quality + pickup time).
        
        Args:
            company_id: ID de l'entreprise
            projected_quality_score: Quality score prévisionnel
            min_pickup_minutes: Temps minimum jusqu'au pickup
            
        Returns:
            True si RL peut être appliqué
        """
        # Garde-fou 1: Bucket routing
        if not self.should_apply_rl(company_id):
            return False
        
        # Garde-fou 2: Quality score
        rl_config = getattr(self.settings, "rl", None)
        min_quality_score = (
            getattr(rl_config, "min_quality_score", 70.0) 
            if rl_config 
            else 70.0
        )
        
        if projected_quality_score < min_quality_score:
            logger.warning(
                "[B4] Company %d: quality_score=%.1f < seuil=%.1f",
                company_id, projected_quality_score, min_quality_score
            )
            return False
        
        # Garde-fou 3: Pickup time minimum
        if min_pickup_minutes is not None:
            min_pickup_threshold = getattr(rl_config, "min_pickup_minutes", 5.0)
            if min_pickup_minutes < min_pickup_threshold:
                logger.warning(
                    "[B4] Company %d: pickup time=%.1fmin < seuil=%.1fmin",
                    company_id, min_pickup_minutes, min_pickup_threshold
                )
                return False
        
        return True
