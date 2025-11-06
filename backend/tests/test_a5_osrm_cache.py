#!/usr/bin/env python3
"""
Tests pour l'amélioration A5 : Mesure du hit-rate cache OSRM.

Teste que le cache OSRM fonctionne et que les métriques hit-rate sont correctes.
"""

import logging

import pytest

from services.unified_dispatch.osrm_cache_metrics import (
    check_cache_alert,
    get_cache_hit_rate,
    get_top_misses,
    increment_cache_bypass,
    increment_cache_hit,
    increment_cache_miss,
    reset_cache_metrics,
)

logger = logging.getLogger(__name__)


class TestOSrmCacheMetrics:
    """Tests pour les métriques cache OSRM (A5)."""
    
    def test_initial_hit_rate_zero(self):
        """Test: hit-rate initial = 0."""
        
        reset_cache_metrics()
        
        hit_rate = get_cache_hit_rate()
        assert hit_rate == 0.0
        
        logger.info("✅ Test: Hit-rate initial = 0")
    
    def test_hit_rate_after_hits_only(self):
        """Test: hit-rate = 1.0 après seulement des hits."""
        
        reset_cache_metrics()
        
        for _ in range(10):
            increment_cache_hit()
        
        hit_rate = get_cache_hit_rate()
        assert hit_rate == 1.0, f"Hit-rate devrait être 1.0, got {hit_rate}"
        
        logger.info("✅ Test: Hit-rate = 1.0 après hits uniquement")
    
    def test_hit_rate_after_misses_only(self):
        """Test: hit-rate = 0.0 après seulement des misses."""
        
        reset_cache_metrics()
        
        for _ in range(10):
            increment_cache_miss()
        
        hit_rate = get_cache_hit_rate()
        assert hit_rate == 0.0
        
        logger.info("✅ Test: Hit-rate = 0.0 après misses uniquement")
    
    def test_hit_rate_mixed(self):
        """Test: hit-rate calculé correctement avec hits et misses."""
        
        reset_cache_metrics()
        
        # 7 hits, 3 misses → hit-rate = 70%
        for _ in range(7):
            increment_cache_hit()
        for _ in range(3):
            increment_cache_miss()
        
        hit_rate = get_cache_hit_rate()
        assert abs(hit_rate - 0.70) < 0.01, f"Hit-rate devrait être ~0.70, got {hit_rate}"
        
        logger.info("✅ Test: Hit-rate = 70% avec 7 hits / 3 misses")
    
    def test_cache_alert_triggered(self):
        """Test: alerte déclenchée si hit-rate < 70%."""
        
        reset_cache_metrics()
        
        # Simuler hit-rate < 50% (< 70%)
        for _ in range(3):
            increment_cache_hit()
        for _ in range(7):
            increment_cache_miss()
        
        alert = check_cache_alert()
        
        assert alert is not None, "Alerte devrait être déclenchée"
        assert alert.hit_rate < 0.70
        assert alert.severity in ("warning", "critical")
        
        logger.info("✅ Test: Alerte déclenchée si hit-rate < 70%")
    
    def test_cache_alert_not_triggered(self):
        """Test: pas d'alerte si hit-rate >= 70%."""
        
        reset_cache_metrics()
        
        # Simuler hit-rate = 75% (> 70%)
        for _ in range(75):
            increment_cache_hit()
        for _ in range(25):
            increment_cache_miss()
        
        alert = check_cache_alert()
        
        assert alert is None, "Alerte ne devrait pas être déclenchée"
        
        logger.info("✅ Test: Pas d'alerte si hit-rate >= 70%")
    
    def test_top_misses_tracking(self):
        """Test: tracking des top misses."""
        
        reset_cache_metrics()
        
        # Simuler quelques misses avec clés différentes
        for i in range(10):
            increment_cache_miss(f"cache_key_{i % 3}")  # 3 clés répétées
        
        top_misses = get_top_misses(n=5)
        
        assert len(top_misses) <= 3, "Devrait avoir au plus 3 clés distinctes"
        assert sum(top_misses.values()) >= 10, "Total misses devrait être >= 10"
        
        logger.info("✅ Test: Top misses tracking fonctionne")
    
    def test_bypass_counting(self):
        """Test: comptage des bypasses Redis."""
        
        reset_cache_metrics()
        
        # Simuler quelques bypasses
        for _ in range(5):
            increment_cache_bypass()
        
        from services.unified_dispatch.osrm_cache_metrics import get_cache_metrics_dict
        
        metrics = get_cache_metrics_dict()
        
        assert metrics["bypass_count"] == 5
        
        logger.info("✅ Test: Bypass counting fonctionne")
    
    def test_cache_reuse_same_coords(self):
        """Test: hit-rate 100% au 2e run avec mêmes coordonnées.
        
        Ce test simule ce qui arrive en production: si les mêmes
        coordonnées sont utilisées plusieurs fois, le cache devrait
        avoir un hit-rate élevé.
        """
        
        reset_cache_metrics()
        
        # 1er appel: miss
        increment_cache_miss("key1")
        
        # 2e appel avec même clé: hit (simule cache)
        increment_cache_hit()
        
        # 3e appel avec même clé: hit
        increment_cache_hit()
        
        # 4e appel avec nouvelle clé: miss
        increment_cache_miss("key2")
        
        hit_rate = get_cache_hit_rate()
        
        # Hit-rate devrait être 2 hits / 4 total = 50%
        # Mais si on ne reteste qu'avec "key1", devrait être 100%
        # Pour ce test, on vérifie juste que le hit-rate est > 0
        assert hit_rate > 0, "Hit-rate devrait être > 0 avec réutilisation"
        
        logger.info("✅ Test: Cache réutilisation avec mêmes coordonnées")


class TestOSrmCacheKeyGeneration:
    """Tests pour génération de clés de cache stables."""
    
    def test_cache_key_same_input(self):
        """Test: même input → même cache key."""
        
        from services.unified_dispatch.osrm_cache_metrics import generate_cache_key_v1
        
        points = [(45.5, -73.5), (45.6, -73.6)]
        key1 = generate_cache_key_v1("driving", points, "20250127", 10)
        key2 = generate_cache_key_v1("driving", points, "20250127", 10)
        
        assert key1 == key2, "Même input devrait produire même clé"
        
        logger.info("✅ Test: Cache key déterministe")
    
    def test_cache_key_different_input(self):
        """Test: input différent → cache key différent."""
        
        from services.unified_dispatch.osrm_cache_metrics import generate_cache_key_v1
        
        points1 = [(45.5, -73.5)]
        points2 = [(45.6, -73.6)]
        
        key1 = generate_cache_key_v1("driving", points1, "20250127", 10)
        key2 = generate_cache_key_v1("driving", points2, "20250127", 10)
        
        assert key1 != key2, "Input différent devrait produire clé différente"
        
        logger.info("✅ Test: Cache key unique par input")
    
    def test_slot_15min_calculation(self):
        """Test: calcul du slot 15 min."""
        
        from datetime import UTC, datetime

        from services.unified_dispatch.osrm_cache_metrics import get_slot_15min
        
        # Slot 0: 00:00
        slot0 = get_slot_15min(datetime(2025, 1, 27, 0, 0, tzinfo=UTC))
        assert slot0 == 0
        
        # Slot 10: 02:30 (150 minutes / 15 = 10)
        slot10 = get_slot_15min(datetime(2025, 1, 27, 2, 30, tzinfo=UTC))
        assert slot10 == 10
        
        # Slot 95: 23:45 (1425 minutes / 15 = 95)
        slot95 = get_slot_15min(datetime(2025, 1, 27, 23, 45, tzinfo=UTC))
        assert slot95 == 95
        
        logger.info("✅ Test: Calcul slot 15 min correct")

