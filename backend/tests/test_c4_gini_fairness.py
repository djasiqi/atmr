#!/usr/bin/env python3
"""
Tests pour C4 : Fairness "Gini" + slider équité/efficacité.

Teste que l'indice Gini est visible et que l'opérateur peut ajuster sans redéploiement.
"""

import logging

import pytest

from services.unified_dispatch.dispatch_metrics import DispatchMetricsCollector

logger = logging.getLogger(__name__)


class TestGiniFairness:
    """Tests pour Fairness Gini (C4)."""

    def test_gini_perfectly_fair(self):
        """Test: Gini = 0 pour distribution parfaitement équitable."""

        # Distribution parfaitement équitable: [5, 5, 5, 5]
        values = [5, 5, 5, 5]

        collector = DispatchMetricsCollector(
            dispatch_run_id=1, company_id=1, date="2025-01-27", all_bookings=[], run_metadata={}
        )

        gini = collector._calculate_gini_index(values)

        assert gini == 0.0, f"Gini devrait être 0 pour distribution parfaite, got {gini}"

        logger.info("✅ Test: Gini = 0 pour distribution parfaitement équitable")

    def test_gini_perfectly_unfair(self):
        """Test: Gini = 1 pour distribution parfaitement inéquitable."""

        # Distribution parfaitement inéquitable: [10, 0, 0, 0] (un seul driver)
        values = [10, 0, 0, 0]

        collector = DispatchMetricsCollector(
            dispatch_run_id=1, company_id=1, date="2025-01-27", all_bookings=[], run_metadata={}
        )

        gini = collector._calculate_gini_index(values)

        # Gini devrait être proche de 1 pour cette distribution
        assert gini > 0.7, f"Gini devrait être élevé (>0.7), got {gini}"

        logger.info("✅ Test: Gini = %.2f pour distribution inéquitable", gini)

    def test_gini_bounds(self):
        """Test: Gini toujours dans [0, 1]."""

        collector = DispatchMetricsCollector(
            dispatch_run_id=1, company_id=1, date="2025-01-27", all_bookings=[], run_metadata={}
        )

        # Tester différentes distributions
        test_cases = [
            [1, 1, 1],  # Parfaitement équitable
            [10, 1, 1],  # Inéquitable
            [5, 4, 3, 2, 1],  # Graduelle
            [10, 0],  # Totalement inéquitable
        ]

        for values in test_cases:
            gini = collector._calculate_gini_index(values)
            assert 0.0 <= gini <= 1.0, f"Gini hors bornes: {gini} pour {values}"

        logger.info("✅ Test: Gini toujours dans [0, 1]")

    def test_slider_effect_on_scores(self):
        """Test: Ajustement slider impacte scoring (via MultiObjectiveSettings)."""

        from services.unified_dispatch.settings import Settings

        # Slider à 50/50 par défaut
        settings = Settings()

        # Vérifier que fairness_weight = 0.5 par défaut
        assert settings.multi_objective.fairness_weight == 0.5
        assert settings.multi_objective.efficiency_weight == 0.5

        # Ajuster slider vers efficacité pure (0.0)
        settings.multi_objective.fairness_weight = 0.0
        assert settings.multi_objective.efficiency_weight == 1.0

        # Ajuster slider vers équité pure (1.0)
        settings.multi_objective.fairness_weight = 1.0
        assert settings.multi_objective.efficiency_weight == 0.0

        logger.info("✅ Test: Slider affecte les poids équité/efficacité")

    def test_pareto_solutions_count(self):
        """Test: Pareto conserve 2-3 solutions candidates."""

        from services.unified_dispatch.settings import Settings

        settings = Settings()

        # Par défaut: 3 solutions Pareto
        assert settings.multi_objective.pareto_solutions_count == 3

        # Ajustable sans redéploiement
        settings.multi_objective.pareto_solutions_count = 2

        assert settings.multi_objective.pareto_solutions_count == 2

        logger.info("✅ Test: Pareto conservent 2-3 solutions candidates")

    def test_realtime_adjustment(self):
        """Test: Ajustement temps réel activé par défaut."""

        from services.unified_dispatch.settings import Settings

        settings = Settings()

        # Par défaut: ajustement temps réel activé
        assert settings.multi_objective.enable_realtime_adjustment is True

        # Peut être désactivé pour debugging
        settings.multi_objective.enable_realtime_adjustment = False

        assert settings.multi_objective.enable_realtime_adjustment is False

        logger.info("✅ Test: Ajustement temps réel activé par défaut")

    def test_fairness_vs_efficiency_tradeoff(self):
        """Test: Tradeoff équité vs efficacité dans scoring."""

        from services.unified_dispatch.settings import Settings

        # Scénario 1: Prioriser efficacité (0.2 équité, 0.8 efficacité)
        settings_efficiency = Settings()
        settings_efficiency.multi_objective.fairness_weight = 0.2

        # Scénario 2: Prioriser équité (0.8 équité, 0.2 efficacité)
        settings_fairness = Settings()
        settings_fairness.multi_objective.fairness_weight = 0.8

        # Vérifier que les poids sont cohérents
        assert settings_efficiency.multi_objective.efficiency_weight == 0.8
        assert settings_fairness.multi_objective.efficiency_weight == 0.2

        logger.info("✅ Test: Tradeoff équité vs efficacité fonctionnel")

    def test_gini_exposed_in_metrics(self):
        """Test: Gini exposé dans DispatchQualityMetrics."""

        from datetime import date, datetime

        from services.unified_dispatch.dispatch_metrics import DispatchQualityMetrics

        # Créer métriques mock
        metrics = DispatchQualityMetrics(
            dispatch_run_id=1,
            company_id=1,
            date=date.today(),
            calculated_at=datetime.now(),
            total_bookings=10,
            assigned_bookings=10,
            unassigned_bookings=0,
            assignment_rate=100.0,
            pooled_bookings=0,
            pooling_rate=0.0,
            on_time_bookings=10,
            delayed_bookings=0,
            average_delay_minutes=0.0,
            max_delay_minutes=0,
            drivers_used=3,
            avg_bookings_per_driver=3.33,
            max_bookings_per_driver=4,
            min_bookings_per_driver=3,
            fairness_coefficient=0.9,
            gini_index=0.15,  # ✅ C4: Indice Gini
            total_distance_km=50.0,
            avg_distance_per_booking=5.0,
            emergency_drivers_used=0,
            emergency_bookings=0,
            solver_used=True,
            heuristic_used=False,
            fallback_used=False,
            execution_time_sec=2.5,
            quality_score=85.0,
        )

        # Vérifier que Gini est exposé
        assert hasattr(metrics, "gini_index")
        assert metrics.gini_index == 0.15

        # Vérifier export JSON
        metrics_dict = metrics.to_dict()
        assert "gini_index" in metrics_dict
        assert metrics_dict["gini_index"] == 0.15

        logger.info("✅ Test: Gini exposé dans DispatchQualityMetrics")

    def test_slider_operation_adjustment(self):
        """Test: Opérateur ajuste sans redéploiement."""

        from services.unified_dispatch.settings import Settings

        settings = Settings()

        # Simulation: Opérateur ajuste slider via UI
        # De 50/50 → 70% équité, 30% efficacité

        old_fairness = settings.multi_objective.fairness_weight
        settings.multi_objective.fairness_weight = 0.7

        # Vérifier changements sans redéploiement
        assert old_fairness != settings.multi_objective.fairness_weight
        assert settings.multi_objective.fairness_weight == 0.7
        assert settings.multi_objective.efficiency_weight == 0.3

        logger.info(
            "✅ Test: Opérateur ajuste slider sans redéploiement (%.1f%% équité)",
            settings.multi_objective.fairness_weight * 100,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
