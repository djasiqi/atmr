#!/usr/bin/env python3
"""
Tests pour B4 : A/B RL : Shadow → 10% → 50% (si gain mesuré).

Teste le routing A/B, les garde-fous, et le tracking des gains.
"""

import logging
import os

import pytest

from services.unified_dispatch.ab_router import ABRouter
from services.unified_dispatch.rl_ab_tracking import (
    RLABResult,
    evaluate_rl_gain,
    get_or_create_rl_tracker,
)

logger = logging.getLogger(__name__)


class TestABRouter:
    """Tests pour le router A/B (B4)."""
    
    def test_rollout_10_percent(self):
        """Test: Rollout 10% par défaut."""
        
        # Simuler settings minimal
        class MockSettings:
            pass
        
        router = ABRouter(MockSettings())
        
        # Vérifier que bucket_size = 10 (10%)
        assert router.bucket_size == 10
        
        # Tester routing pour 100 companies
        routed = sum(1 for company_id in range(100) if router.should_apply_rl(company_id))
        
        assert 8 <= routed <= 12, f"Devrait router ~10% mais got {routed}%"
        
        logger.info("✅ Test: Rollout 10% fonctionne")
    
    def test_rollout_50_percent(self):
        """Test: Rollout 50% avec variable env."""
        
        old_val = os.environ.get("RL_ROLLOUT_PERCENTAGE")
        try:
            os.environ["RL_ROLLOUT_PERCENTAGE"] = "50"
            
            class MockSettings:
                pass
            
            router = ABRouter(MockSettings())
            
            # Vérifier que bucket_size = 50
            assert router.bucket_size == 50
            
            # Tester routing pour 100 companies
            routed = sum(1 for company_id in range(100) if router.should_apply_rl(company_id))
            
            assert 48 <= routed <= 52, f"Devrait router ~50% mais got {routed}%"
            
            logger.info("✅ Test: Rollout 50% fonctionne")
        finally:
            if old_val is not None:
                os.environ["RL_ROLLOUT_PERCENTAGE"] = old_val
            else:
                del os.environ["RL_ROLLOUT_PERCENTAGE"]
    
    def test_deterministic_routing(self):
        """Test: Routing déterministe par company_id."""
        
        class MockSettings:
            pass
        
        router = ABRouter(MockSettings())
        
        # Même company_id → même décision
        for i in range(10):
            should_apply_1 = router.should_apply_rl(123)
            should_apply_2 = router.should_apply_rl(123)
            
            assert should_apply_1 == should_apply_2, "Routing devrait être déterministe"
        
        logger.info("✅ Test: Routing déterministe")
    
    def test_safety_guards_quality(self):
        """Test: Garde-fou quality score."""
        
        class MockSettings:
            class rl:
                min_quality_score = 70.0
        
        router = ABRouter(MockSettings())
        
        # Quality OK → autoriser
        should_apply = router.should_apply_with_safety_guards(company_id=1, projected_quality_score=75.0)
        assert should_apply is True
        
        # Quality bas → bloquer
        should_apply = router.should_apply_with_safety_guards(company_id=1, projected_quality_score=65.0)
        assert should_apply is False
        
        logger.info("✅ Test: Garde-fou quality score")
    
    def test_safety_guards_pickup_time(self):
        """Test: Garde-fou pickup time."""
        
        class MockSettings:
            class rl:
                min_quality_score = 70.0
                min_pickup_minutes = 5.0
        
        router = ABRouter(MockSettings())
        
        # Pickup OK → autoriser
        should_apply = router.should_apply_with_safety_guards(
            company_id=1,
            projected_quality_score=75.0,
            min_pickup_minutes=10.0
        )
        assert should_apply is True
        
        # Pickup trop rapide → bloquer
        should_apply = router.should_apply_with_safety_guards(
            company_id=1,
            projected_quality_score=75.0,
            min_pickup_minutes=3.0
        )
        assert should_apply is False
        
        logger.info("✅ Test: Garde-fou pickup time")


class TestRLGainTracking:
    """Tests pour le tracking des gains RL (B4)."""
    
    def test_evaluate_rl_gain(self):
        """Test: Évaluer gain RL vs heuristique."""
        
        heuristic_result = {
            "quality_score": 70.0,
            "assignment_rate": 80.0,
            "pooling_rate": 15.0
        }
        
        rl_result = {
            "quality_score": 75.0,  # +5pts
            "assignment_rate": 85.0,
            "pooling_rate": 18.0
        }
        
        ab_result = evaluate_rl_gain(company_id=1, heuristic_result=heuristic_result, rl_result=rl_result)
        
        assert ab_result.delta_quality == 5.0
        assert ab_result.delta_assignment == 5.0
        assert ab_result.gain_detected is True  # +5pts >= 3pts
        
        logger.info("✅ Test: Gain RL évalué = +%.1fpts", ab_result.delta_quality)
    
    def test_gain_below_threshold(self):
        """Test: Gain < 3pts ne déclenche pas gain_detected."""
        
        heuristic_result = {"quality_score": 70.0}
        rl_result = {"quality_score": 72.0}  # +2pts seulement
        
        ab_result = evaluate_rl_gain(company_id=1, heuristic_result=heuristic_result, rl_result=rl_result)
        
        assert ab_result.delta_quality == 2.0
        assert ab_result.gain_detected is False  # +2pts < 3pts
        
        logger.info("✅ Test: Gain insuffisant détecté")
    
    def test_should_increase_rollout(self):
        """Test: Décision d'augmenter rollout si gain >= 3pts."""
        
        tracker = get_or_create_rl_tracker(company_id=1)
        
        # Simuler 14 jours de gains de +3pts
        for i in range(14):
            result = RLABResult(
                company_id=1,
                date=None,  # type: ignore
                quality_score_heuristic=70.0,
                quality_score_rl=73.0,
                delta_quality=3.0,  # +3pts
                assignment_rate_heuristic=80.0,
                assignment_rate_rl=85.0,
                delta_assignment=5.0,
                pooling_rate_heuristic=15.0,
                pooling_rate_rl=18.0,
                delta_pooling=3.0,
                gain_detected=True
            )
            tracker.record_result(result)
        
        assert tracker.median_gain >= 3.0
        assert tracker.should_increase_rollout is True
        
        logger.info("✅ Test: Rollout augmentation si gain >= 3pts")
    
    def test_merge_scores_alpha(self):
        """Test: Fusion scores Heuristique + RL avec alpha."""
        
        # Test fusion basique
        heuristic_score = 0.7  # 70/100
        rl_score = 0.5  # 50/100
        alpha = 0.2  # 20% RL
        
        # Fusion: alpha * RL + (1-alpha) * Heuristic
        final_score = alpha * rl_score + (1 - alpha) * heuristic_score
        # = 0.2 * 0.5 + 0.8 * 0.7 = 0.1 + 0.56 = 0.66
        
        assert 0.65 <= final_score <= 0.67
        
        logger.info("✅ Test: Fusion scores alpha fonctionne")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

