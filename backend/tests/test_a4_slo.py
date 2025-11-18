#!/usr/bin/env python3
"""
Tests pour l'amélioration A4 : SLO déclarés + métriques Prometheus.

Teste que les SLO sont respectés pour différentes tailles de batch.
"""

import logging
import time
from datetime import UTC, datetime

import pytest

from services.unified_dispatch.slo import (
    SLOBreachTracker,
    check_slo_breach,
    get_slo_for_batch_size,
    reset_slo_tracker,
)

logger = logging.getLogger(__name__)


class TestSLO:
    """Tests pour les SLO (A4)."""

    def test_slo_small_batch(self):
        """Test: SLO pour petits batches (< 50 bookings)."""

        slo = get_slo_for_batch_size(30)

        assert slo.latency_p95_max_ms == 10000  # 10s
        assert slo.success_rate_min == 0.95  # 95%
        assert slo.quality_score_min == 80.0

        logger.info("✅ Test: SLO petit batch correct")

    def test_slo_medium_batch(self):
        """Test: SLO pour batches moyens (50-200 bookings)."""

        slo = get_slo_for_batch_size(150)

        assert slo.latency_p95_max_ms == 30000  # 30s
        assert slo.success_rate_min == 0.90  # 90%
        assert slo.quality_score_min == 75.0

        logger.info("✅ Test: SLO batch moyen correct")

    def test_slo_large_batch(self):
        """Test: SLO pour grands batches (> 200 bookings)."""

        slo = get_slo_for_batch_size(500)

        assert slo.latency_p95_max_ms == 60000  # 60s
        assert slo.success_rate_min == 0.85  # 85%
        assert slo.quality_score_min == 70.0

        logger.info("✅ Test: SLO grand batch correct")

    def test_slo_check_no_breach(self):
        """Test: SLO check sans breach."""

        # Simuler des métriques qui respectent les SLO
        result = check_slo_breach(
            total_time_sec=8.0,  # 8s < 10s (p95 small)
            assignment_rate=0.97,  # 97% > 95% min
            quality_score=85.0,  # 85 > 80 min
            n_bookings=30,  # small batch
        )

        assert result["breached"] is False
        assert len(result["breaches"]) == 0

        logger.info("✅ Test: SLO check sans breach")

    def test_slo_check_latency_breach(self):
        """Test: SLO check avec breach latence."""

        # Latence trop élevée
        result = check_slo_breach(
            total_time_sec=12.0,  # 12s > 10s (p95 small) → breach
            assignment_rate=0.98,  # OK
            quality_score=85.0,  # OK
            n_bookings=30,
        )

        assert result["breached"] is True
        assert result["latency_breach"] is True
        assert len([b for b in result["breaches"] if b["dimension"] == "latency"]) == 1

        logger.info("✅ Test: SLO breach latence détectée")

    def test_slo_check_success_breach(self):
        """Test: SLO check avec breach success rate."""

        # Success rate trop faible
        result = check_slo_breach(
            total_time_sec=8.0,  # OK
            assignment_rate=0.88,  # 88% < 95% min → breach
            quality_score=85.0,  # OK
            n_bookings=30,
        )

        assert result["breached"] is True
        assert result["success_breach"] is True
        assert len([b for b in result["breaches"] if b["dimension"] == "success_rate"]) == 1

        logger.info("✅ Test: SLO breach success rate détectée")

    def test_slo_check_quality_breach(self):
        """Test: SLO check avec breach quality score."""

        # Quality score trop faible
        result = check_slo_breach(
            total_time_sec=8.0,  # OK
            assignment_rate=0.98,  # OK
            quality_score=70.0,  # 70 < 80 min → breach
            n_bookings=30,
        )

        assert result["breached"] is True
        assert result["quality_breach"] is True
        assert len([b for b in result["breaches"] if b["dimension"] == "quality_score"]) == 1

        logger.info("✅ Test: SLO breach quality score détectée")

    def test_slo_tracker_should_alert(self):
        """Test: SLO tracker alerte après 3 breaches."""

        tracker = SLOBreachTracker(window_minutes=15, breach_threshold=3)
        current_time = time.time()

        # Enregistrer 3 breaches
        tracker.record_breach("latency", current_time - 100)
        tracker.record_breach("success_rate", current_time - 200)
        tracker.record_breach("quality_score", current_time - 300)

        # Vérifier que should_alert = True
        assert tracker.should_alert(current_time) is True

        # Vérifier le résumé
        summary = tracker.get_breach_summary(current_time)
        assert summary["breach_count"] == 3
        assert summary["should_alert"] is True
        assert summary["severity"] == "warning"

        logger.info("✅ Test: SLO tracker alerte après 3 breaches")

    def test_slo_tracker_no_alert(self):
        """Test: SLO tracker pas d'alerte si < 3 breaches."""

        tracker = SLOBreachTracker(window_minutes=15, breach_threshold=3)
        current_time = time.time()

        # Enregistrer 2 breaches seulement
        tracker.record_breach("latency", current_time - 100)
        tracker.record_breach("success_rate", current_time - 200)

        # Vérifier que should_alert = False
        assert tracker.should_alert(current_time) is False

        logger.info("✅ Test: SLO tracker pas d'alerte si < 3 breaches")

    def test_slo_tracker_window_expiry(self):
        """Test: SLO tracker oublie les breaches anciennes."""

        tracker = SLOBreachTracker(window_minutes=1, breach_threshold=3)  # Fenêtre 1 min
        current_time = time.time()

        # Enregistrer 3 breaches, mais 2 sont trop anciennes (> 1 min)
        tracker.record_breach("latency", current_time - 120)  # > 1 min
        tracker.record_breach("success_rate", current_time - 130)  # > 1 min
        tracker.record_breach("quality_score", current_time - 10)  # < 1 min

        # Vérifier que should_alert = False (seulement 1 breach dans fenêtre)
        assert tracker.should_alert(current_time) is False

        summary = tracker.get_breach_summary(current_time)
        assert summary["breach_count"] == 1

        logger.info("✅ Test: SLO tracker fenêtre temporelle fonctionne")

    def test_slo_budget_50(self):
        """Test: SLO budget 50 bookings respecté."""

        # Simuler un dispatch de 50 bookings qui respecte les SLO medium
        result = check_slo_breach(
            total_time_sec=25.0,  # 25s < 30s (p95 medium)
            assignment_rate=0.92,  # 92% > 90% min
            quality_score=78.0,  # 78 > 75 min
            n_bookings=50,
        )

        assert result["breached"] is False

        logger.info("✅ Test: SLO budget 50 bookings respecté")

    def test_slo_budget_200(self):
        """Test: SLO budget 200 bookings respecté."""

        # Simuler un dispatch de 200 bookings qui respecte les SLO large
        result = check_slo_breach(
            total_time_sec=55.0,  # 55s < 60s (p95 large)
            assignment_rate=0.88,  # 88% > 85% min
            quality_score=72.0,  # 72 > 70 min
            n_bookings=200,
        )

        assert result["breached"] is False

        logger.info("✅ Test: SLO budget 200 bookings respecté")

    def test_slo_budget_200_breach(self):
        """Test: SLO budget 200 bookings avec breach latence."""

        # Simuler un dispatch de 200 bookings avec latence excessive
        result = check_slo_breach(
            total_time_sec=70.0,  # 70s > 60s (p95 large) → breach
            assignment_rate=0.90,  # OK
            quality_score=75.0,  # OK
            n_bookings=200,
        )

        assert result["breached"] is True
        assert result["latency_breach"] is True

        logger.info("✅ Test: SLO budget 200 bookings avec breach")
