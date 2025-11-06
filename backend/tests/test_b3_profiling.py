#!/usr/bin/env python3
"""
Tests pour B3 : Profiling par fonction & budgets de perf.

Teste que le profiling fonctionne, que les budgets sont respectés,
et que les rapports sont générés correctement.
"""

import logging
import os

import pytest

from scripts.profiling.profiler import (
    PERFORMANCE_BUDGETS,
    DispatchProfiler,
    get_profiler,
    is_profiling_enabled,
)

logger = logging.getLogger(__name__)


class TestDispatchProfiler:
    """Tests pour le système de profiling (B3)."""
    
    def test_profiler_disabled_by_default(self):
        """Test: Profiling désactivé par défaut."""
        
        # Simuler variable d'environnement non définie
        old_val = os.environ.get("ENABLE_PROFILING")
        try:
            if "ENABLE_PROFILING" in os.environ:
                del os.environ["ENABLE_PROFILING"]
            
            profiler = get_profiler()
            assert not profiler.is_enabled()
        finally:
            if old_val is not None:
                os.environ["ENABLE_PROFILING"] = old_val
        
        logger.info("✅ Test: Profiling désactivé par défaut")
    
    def test_profiler_enabled_with_env(self):
        """Test: Profiling activé avec ENABLE_PROFILING=1."""
        
        old_val = os.environ.get("ENABLE_PROFILING")
        try:
            os.environ["ENABLE_PROFILING"] = "1"
            
            profiler = get_profiler()
            assert profiler.is_enabled()
        finally:
            if old_val is not None:
                os.environ["ENABLE_PROFILING"] = old_val
            else:
                del os.environ["ENABLE_PROFILING"]
        
        logger.info("✅ Test: Profiling activé avec ENABLE_PROFILING=1")
    
    def test_performance_budgets_defined(self):
        """Test: Vérifier que les budgets sont définis."""
        
        assert "data_collection" in PERFORMANCE_BUDGETS
        assert "heuristics" in PERFORMANCE_BUDGETS
        assert "solver" in PERFORMANCE_BUDGETS
        assert "persistence" in PERFORMANCE_BUDGETS
        assert "total" in PERFORMANCE_BUDGETS
        
        # Vérifier que les budgets sont > 0
        for budget in PERFORMANCE_BUDGETS.values():
            assert budget > 0
        
        logger.info("✅ Test: Budgets de performance définis")
    
    def test_budget_check_all_respected(self):
        """Test: Check budgets si tous sont respectés."""
        
        from scripts.profiling.profiler import DispatchProfiler
        from services.unified_dispatch.performance_metrics import DispatchPerformanceMetrics
        
        profiler = DispatchProfiler(enabled=False)
        
        # Créer métriques hypothétiques avec temps respectant budgets
        metrics = DispatchPerformanceMetrics(
            data_collection_time=5.0,  # 5s < 10s budget
            heuristics_time=10.0,  # 10s < 15s budget
            solver_time=20.0,  # 20s < 30s budget
            persistence_time=2.0,  # 2s < 5s budget
            total_time=37.0,  # 37s < 60s budget
        )
        
        result = profiler.check_budgets(metrics)
        
        assert result["all_respected"] is True
        assert len(result["issues"]) == 0
        
        logger.info("✅ Test: Tous les budgets respectés")
    
    def test_budget_check_exceeded(self):
        """Test: Check budgets si certains sont dépassés."""
        
        from scripts.profiling.profiler import DispatchProfiler
        from services.unified_dispatch.performance_metrics import DispatchPerformanceMetrics
        
        profiler = DispatchProfiler(enabled=False)
        
        # Créer métriques avec solver dépassé
        metrics = DispatchPerformanceMetrics(
            data_collection_time=8.0,  # OK
            heuristics_time=12.0,  # OK
            solver_time=35.0,  # EXCEEDED (35s > 30s budget)
            persistence_time=3.0,  # OK
            total_time=58.0,  # OK
        )
        
        result = profiler.check_budgets(metrics)
        
        assert result["all_respected"] is False
        assert len(result["issues"]) == 1
        assert result["issues"][0]["stage"] == "solver"
        
        logger.info("✅ Test: Budget solver dépassé détecté")
    
    def test_profile_stage_times(self):
        """Test: Vérifier que les temps de stage sont trackés."""
        
        profiler = DispatchProfiler(enabled=False)
        
        # Simuler quelques stages
        with profiler.profile_stage("test_stage_1"):
            import time
            time.sleep(0.01)  # 10ms
        
        with profiler.profile_stage("test_stage_2"):
            time.sleep(0.02)  # 20ms
        
        assert "test_stage_1" in profiler.function_times
        assert "test_stage_2" in profiler.function_times
        
        # Vérifier que les temps sont > 0
        assert profiler.function_times["test_stage_1"] > 0
        assert profiler.function_times["test_stage_2"] > profiler.function_times["test_stage_1"]
        
        logger.info("✅ Test: Temps de stages trackés")
    
    def test_profile_report_generation(self):
        """Test: Vérifier que le rapport est généré."""
        
        from scripts.profiling.profiler import DispatchProfiler
        from services.unified_dispatch.performance_metrics import DispatchPerformanceMetrics
        
        profiler = DispatchProfiler(enabled=False)
        
        # Créer métriques
        metrics = DispatchPerformanceMetrics(
            data_collection_time=5.0,
            heuristics_time=10.0,
            solver_time=20.0,
            persistence_time=2.0,
            total_time=37.0,
        )
        
        report = profiler.generate_report(metrics)
        
        assert "B3 PROFILING REPORT" in report
        assert "PERFORMANCE BUDGETS" in report
        assert "data_collection" in report
        assert "heuristics" in report
        
        logger.info("✅ Test: Rapport de profiling généré")
    
    def test_profile_enabled_profiling(self):
        """Test: Profiling avec cProfile activé (smoke test)."""
        
        profiler = DispatchProfiler(enabled=True)
        
        assert profiler.is_enabled()
        assert profiler.profiler is not None
        
        # Démarrer/arrêter
        profiler.start()
        
        # Simuler quelque chose à profiler
        def dummy_function():
            return sum(range(1000))
        
        for _ in range(100):
            dummy_function()
        
        profiler.stop()
        
        # Vérifier qu'on peut récupérer les top fonctions
        top_funcs = profiler.get_top_functions(n=5)
        
        assert isinstance(top_funcs, list)
        
        logger.info("✅ Test: Profiling cProfile fonctionne")
    
    def test_budget_percentages(self):
        """Test: Vérifier que les pourcentages sont calculés correctement."""
        
        from scripts.profiling.profiler import DispatchProfiler
        from services.unified_dispatch.performance_metrics import DispatchPerformanceMetrics
        
        profiler = DispatchProfiler(enabled=False)
        
        # Créer métriques à 50% des budgets
        metrics = DispatchPerformanceMetrics(
            data_collection_time=5.0,  # 50% de 10s
            heuristics_time=7.5,  # 50% de 15s
            solver_time=15.0,  # 50% de 30s
            persistence_time=2.5,  # 50% de 5s
            total_time=30.0,  # 50% de 60s
        )
        
        result = profiler.check_budgets(metrics)
        
        for stage, info in result["budgets"].items():
            assert 45 <= info["pct_of_budget"] <= 55, f"{stage} devrait être ~50%"
        
        logger.info("✅ Test: Pourcentages de budgets calculés")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

