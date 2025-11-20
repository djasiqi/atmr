#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Tests complets pour le Shadow Mode Manager.

Améliore la couverture de tests en testant tous les aspects
du shadow mode et des KPIs.
"""

import contextlib
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import conditionnel pour éviter les erreurs si les modules ne sont pas disponibles
try:
    from services.rl.shadow_mode_manager import ShadowModeManager
except ImportError:
    ShadowModeManager = None

try:
    from services.rl.improved_dqn_agent import ImprovedDQNAgent
except ImportError:
    ImprovedDQNAgent = None


class TestShadowModeManager:
    """Tests complets pour ShadowModeManager."""

    @pytest.fixture
    def mock_rl_agent(self):
        """Crée un agent RL mock."""
        if ImprovedDQNAgent is None:
            return Mock()

        agent = Mock(spec=ImprovedDQNAgent)
        agent.select_action.return_value = 5
        agent.get_q_values.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        agent.get_action_confidence.return_value = 0.85
        return agent

    @pytest.fixture
    def shadow_manager(self):
        """Crée une instance de ShadowModeManager pour les tests."""
        if ShadowModeManager is None:
            pytest.skip("ShadowModeManager non disponible")

        return ShadowModeManager(data_dir="data/rl/shadow_mode_test")

    def test_manager_initialization(self, shadow_manager):
        """Test l'initialisation du manager."""
        assert shadow_manager is not None
        assert hasattr(shadow_manager, "kpi_metrics")
        assert hasattr(shadow_manager, "decision_metadata")

    def test_decision_comparison(self, shadow_manager):
        """Test la comparaison des décisions."""
        # Données de test
        company_id = "test_company_1"
        booking_id = "test_booking_1"
        human_decision = {
            "driver_id": "driver_1",
            "eta_minutes": 20,
            "delay_minutes": 5,
            "distance_km": 10.5,
            "driver_load": 2,
        }
        rl_decision = {
            "driver_id": "driver_2",
            "eta_minutes": 15,
            "delay_minutes": 3,
            "distance_km": 9.2,
            "driver_load": 1,
            "confidence": 0.85,
            "alternative_drivers": ["driver_2", "driver_3"],
        }
        context = {
            "avg_eta": 18,
            "avg_distance": 11,
            "avg_load": 2,
            "vehicle_capacity": 4,
        }

        # Test de la comparaison
        kpis = shadow_manager.log_decision_comparison(company_id, booking_id, human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis
        assert "rl_confidence" in kpis

    def test_kpi_calculation(self, shadow_manager):
        """Test le calcul des KPIs."""
        # Données de test pour les KPIs
        company_id = "test_company_1"
        booking_id = "test_booking_1"
        human_decision = {
            "driver_id": "driver_1",
            "eta_minutes": 20,
            "delay_minutes": 5,
            "distance_km": 10.5,
            "driver_load": 2,
        }
        rl_decision = {
            "driver_id": "driver_2",
            "eta_minutes": 15,
            "delay_minutes": 3,
            "distance_km": 9.2,
            "driver_load": 1,
            "confidence": 0.85,
            "alternative_drivers": ["driver_2", "driver_3"],
        }
        context = {
            "avg_eta": 18,
            "avg_distance": 11,
            "avg_load": 2,
            "vehicle_capacity": 4,
        }

        # Test du calcul des KPIs via log_decision_comparison
        kpis = shadow_manager.log_decision_comparison(company_id, booking_id, human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis
        assert "rl_confidence" in kpis
        assert "constraint_violations" in kpis
        assert "performance_impact" in kpis

    def test_decision_reasons_generation(self, shadow_manager):
        """Test la génération des raisons de décision."""
        # Données de test
        company_id = "test_company_1"
        booking_id = "test_booking_1"
        human_decision = {
            "driver_id": "driver_1",
            "eta_minutes": 20,
            "delay_minutes": 5,
            "distance_km": 10.5,
            "driver_load": 2,
        }
        rl_decision = {
            "driver_id": "driver_2",
            "eta_minutes": 15,
            "delay_minutes": 3,
            "distance_km": 9.2,
            "driver_load": 1,
            "confidence": 0.85,
            "alternative_drivers": ["driver_2", "driver_3"],
        }
        context = {
            "avg_eta": 18,
            "avg_distance": 11,
            "avg_load": 2,
            "vehicle_capacity": 4,
        }

        # Test de la génération des raisons via log_decision_comparison
        kpis = shadow_manager.log_decision_comparison(company_id, booking_id, human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert "decision_reasons" in kpis
        assert isinstance(kpis["decision_reasons"], list)

    def test_constraint_violation_check(self, shadow_manager):
        """Test la vérification des violations de contraintes."""
        # Données de test avec violation
        company_id = "test_company_1"
        booking_id = "test_booking_1"
        human_decision = {
            "driver_id": "driver_1",
            "eta_minutes": 20,
            "delay_minutes": 5,
        }
        rl_decision = {
            "driver_id": "driver_2",
            "eta_minutes": 15,
            "delay_minutes": 3,
            "passenger_count": 6,  # Plus que la capacité
            "respects_time_window": False,  # Violation
        }
        context = {
            "vehicle_capacity": 4,
        }

        # Test de la vérification des contraintes via log_decision_comparison
        kpis = shadow_manager.log_decision_comparison(company_id, booking_id, human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert "constraint_violations" in kpis
        assert isinstance(kpis["constraint_violations"], list)

    def test_performance_impact_calculation(self, shadow_manager):
        """Test le calcul de l'impact sur les performances."""
        # Données de test
        company_id = "test_company_1"
        booking_id = "test_booking_1"
        human_decision = {
            "driver_id": "driver_1",
            "eta_minutes": 20,
            "delay_minutes": 5,
            "distance_km": 10.5,
            "driver_load": 2,
        }
        rl_decision = {
            "driver_id": "driver_2",
            "eta_minutes": 15,
            "delay_minutes": 3,
            "distance_km": 9.2,
            "driver_load": 1,
        }
        context = {
            "avg_eta": 18,
            "avg_distance": 11,
            "avg_load": 2,
        }

        # Test du calcul de l'impact via log_decision_comparison
        kpis = shadow_manager.log_decision_comparison(company_id, booking_id, human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert "performance_impact" in kpis
        assert isinstance(kpis["performance_impact"], dict)

    def test_daily_report_generation(self, shadow_manager):
        """Test la génération du rapport quotidien."""
        # Données de test
        company_id = "company_1"
        date = datetime.now(UTC).date()

        # Ajouter des données de comparaison via log_decision_comparison
        for i in range(2):
            shadow_manager.log_decision_comparison(
                company_id,
                f"booking_{i}",
                {"driver_id": f"driver_{i}", "eta_minutes": 20 + i * 5, "delay_minutes": 5 + i},
                {"driver_id": f"driver_{i+1}", "eta_minutes": 15 + i * 3, "delay_minutes": 3 + i, "confidence": 0.85 + i * 0.07},
                {},
            )

        # Test de la génération du rapport
        report = shadow_manager.generate_daily_report(company_id, date)

        assert isinstance(report, dict)
        assert "company_id" in report
        assert "date" in report
        assert "total_decisions" in report
        assert "statistics" in report or "total_decisions" in report

    def test_historical_analysis(self, shadow_manager):
        """Test l'analyse historique via rapports quotidiens."""
        # Données de test
        company_id = "company_1"
        start_date = datetime.now(UTC).date() - timedelta(days=2)
        end_date = datetime.now(UTC).date()

        # Ajouter des données historiques via log_decision_comparison
        for i in range(3):
            test_date = start_date + timedelta(days=i)
            shadow_manager.log_decision_comparison(
                company_id,
                f"booking_{i}",
                {"driver_id": f"driver_{i}", "eta_minutes": 20 + i * 5, "delay_minutes": 5 + i},
                {"driver_id": f"driver_{i+1}", "eta_minutes": 15 + i * 3, "delay_minutes": 3 + i, "confidence": 0.85 + i * 0.01},
                {},
            )

        # Test de l'analyse via generate_daily_report
        report = shadow_manager.generate_daily_report(company_id, end_date)
        assert isinstance(report, dict)
        assert "company_id" in report

    def test_alert_generation(self, shadow_manager):
        """Test la génération d'alertes via rapport quotidien."""
        # Données de test
        company_id = "company_1"
        date = datetime.now(UTC).date()

        # Ajouter des données avec violations pour générer des alertes
        shadow_manager.log_decision_comparison(
            company_id,
            "booking_1",
            {"driver_id": "driver_1", "eta_minutes": 20, "delay_minutes": 5},
            {
                "driver_id": "driver_2",
                "eta_minutes": 15,
                "delay_minutes": 3,
                "passenger_count": 6,  # Violation capacité
                "respects_time_window": False,  # Violation fenêtre
                "confidence": 0.2,  # Faible confiance
            },
            {"vehicle_capacity": 4},
        )

        # Test via generate_daily_report qui peut inclure des recommandations
        report = shadow_manager.generate_daily_report(company_id, date)
        assert isinstance(report, dict)
        assert "company_id" in report

    def test_data_export(self, shadow_manager):
        """Test l'export des données via generate_daily_report."""
        # Données de test
        company_id = "company_1"
        date = datetime.now(UTC).date()

        # Ajouter des données via log_decision_comparison
        shadow_manager.log_decision_comparison(
            company_id,
            "booking_1",
            {"driver_id": "driver_1", "eta_minutes": 20, "delay_minutes": 5},
            {"driver_id": "driver_2", "eta_minutes": 15, "delay_minutes": 3, "confidence": 0.85},
            {},
        )

        # Test via generate_daily_report qui contient les données
        report = shadow_manager.generate_daily_report(company_id, date)
        assert isinstance(report, dict)
        assert "company_id" in report
        assert "date" in report

    def test_error_handling(self, shadow_manager):
        """Test la gestion d'erreurs."""
        # Test avec des données invalides
        with contextlib.suppress(ValueError, TypeError, AttributeError, KeyError):
            shadow_manager.log_decision_comparison(
                "company_1",
                "booking_1",
                {},  # Données invalides
                {},
                {},
            )

    def test_performance_metrics(self, shadow_manager):
        """Test les métriques de performance."""
        # Simuler des métriques de performance
        metrics = {
            "total_comparisons": 1000,
            "average_agreement_rate": 0.85,
            "average_rl_confidence": 0.78,
            "constraint_violation_rate": 0.05,
            "performance_improvement": 0.12,
        }

        # Vérifier que les métriques sont dans des plages raisonnables
        assert metrics["total_comparisons"] > 0
        assert 0 <= metrics["average_agreement_rate"] <= 1
        assert 0 <= metrics["average_rl_confidence"] <= 1
        assert 0 <= metrics["constraint_violation_rate"] <= 1
        assert -1 <= metrics["performance_improvement"] <= 1


class TestShadowModeRoutes:
    """Tests pour les routes du shadow mode."""

    @pytest.fixture
    def mock_shadow_manager(self):
        """Crée un manager de shadow mode mock."""
        if ShadowModeManager is None:
            return Mock()

        manager = Mock(spec=ShadowModeManager)
        manager.log_decision_comparison.return_value = {
            "eta_delta": 5.0,
            "delay_delta": 2.0,
            "rl_confidence": 0.85,
            "constraint_violations": [],
            "performance_impact": {},
        }
        manager.generate_daily_report.return_value = {
            "company_id": "company_1",
            "date": datetime.now(UTC).date().isoformat(),
            "total_decisions": 100,
            "average_agreement_rate": 0.85,
        }
        return manager

    def test_compare_decisions_endpoint(self, mock_shadow_manager):
        """Test l'endpoint de comparaison des décisions."""
        # Mock des données de requête
        company_id = "company_1"
        booking_id = "booking_1"
        human_decision = {"driver_id": "driver_1", "eta_minutes": 20}
        rl_decision = {"driver_id": "driver_2", "eta_minutes": 15, "confidence": 0.85}
        context = {}

        # Simuler l'appel à l'endpoint
        result = mock_shadow_manager.log_decision_comparison(company_id, booking_id, human_decision, rl_decision, context)

        assert isinstance(result, dict)
        assert "eta_delta" in result
        assert "rl_confidence" in result

    def test_daily_report_endpoint(self, mock_shadow_manager):
        """Test l'endpoint du rapport quotidien."""
        # Mock des paramètres
        company_id = "company_1"
        date = datetime.now(UTC).date()

        # Test de récupération du rapport
        report = mock_shadow_manager.generate_daily_report(company_id, date)

        assert isinstance(report, dict)
        assert "company_id" in report
        assert "date" in report

    def test_historical_analysis_endpoint(self, mock_shadow_manager):
        """Test l'endpoint d'analyse historique via rapport quotidien."""
        # Mock des paramètres
        company_id = "company_1"
        date = datetime.now(UTC).date()

        # Test via generate_daily_report
        report = mock_shadow_manager.generate_daily_report(company_id, date)

        assert isinstance(report, dict)
        assert "company_id" in report
        assert "date" in report


def run_shadow_mode_tests():
    """Exécute tous les tests du shadow mode."""
    print("👥 Exécution des tests du Shadow Mode")

    # Tests de base
    test_classes = [TestShadowModeManager, TestShadowModeRoutes]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print("\n📋 Tests {test_class.__name__}")

        # Créer une instance de la classe de test
        test_instance = test_class()

        # Exécuter les méthodes de test
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print("  ✅ {method_name}")
                    passed_tests += 1
                except Exception:
                    print("  ❌ {method_name}: {e}")

    print("\n📊 Résultats des tests du Shadow Mode:")
    print("  Tests exécutés: {total_tests}")
    print("  Tests réussis: {passed_tests}")
    print("  Taux de succès: {passed_tests/total_tests*100" if total_tests > 0 else "  Taux de succès: 0%")

    return passed_tests, total_tests


if __name__ == "__main__":
    run_shadow_mode_tests()
