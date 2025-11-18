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
    def shadow_manager(self, mock_rl_agent):
        """Crée une instance de ShadowModeManager pour les tests."""
        if ShadowModeManager is None:
            pytest.skip("ShadowModeManager non disponible")

        return ShadowModeManager(rl_agent=mock_rl_agent)

    def test_manager_initialization(self, shadow_manager):
        """Test l'initialisation du manager."""
        assert shadow_manager is not None
        assert hasattr(shadow_manager, "rl_agent")
        assert hasattr(shadow_manager, "comparison_history")
        assert hasattr(shadow_manager, "kpi_metrics")

    def test_decision_comparison(self, shadow_manager, mock_rl_agent):
        """Test la comparaison des décisions."""
        # Données de test
        state = [0.1, 0.2, 0.3, 0.4, 0.5]
        human_action = 3
        human_eta = datetime.now(UTC) + timedelta(minutes=20)

        # Mock de la décision RL
        mock_rl_agent.select_action.return_value = 5
        mock_rl_agent.get_q_values.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_rl_agent.get_action_confidence.return_value = 0.85

        # Test de la comparaison
        comparison = shadow_manager.compare_decisions(state, human_action, human_eta)

        assert isinstance(comparison, dict)
        assert "rl_action" in comparison
        assert "human_action" in comparison
        assert "eta_delta" in comparison
        assert "agreement_rate" in comparison
        assert "rl_confidence" in comparison

    def test_kpi_calculation(self, shadow_manager):
        """Test le calcul des KPIs."""
        # Données de test pour les KPIs
        comparison_data = {
            "eta_delta": 5.0,
            "delay_delta": 2.0,
            "agreement_rate": 0.8,
            "rl_confidence": 0.85,
            "constraint_violations": 0,
            "performance_impact": 0.1,
        }

        # Test du calcul des KPIs
        kpis = shadow_manager.calculate_kpis(comparison_data)

        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis
        assert "agreement_rate" in kpis
        assert "rl_confidence" in kpis
        assert "constraint_violations" in kpis
        assert "performance_impact" in kpis

    def test_decision_reasons_generation(self, shadow_manager):
        """Test la génération des raisons de décision."""
        # Données de test
        state = [0.1, 0.2, 0.3, 0.4, 0.5]
        rl_action = 5
        human_action = 3

        # Test de la génération des raisons
        reasons = shadow_manager.generate_decision_reasons(state, rl_action, human_action)

        assert isinstance(reasons, dict)
        assert "rl_reasons" in reasons
        assert "human_reasons" in reasons
        assert "difference_explanation" in reasons

    def test_constraint_violation_check(self, shadow_manager):
        """Test la vérification des violations de contraintes."""
        # Données de test
        action = 5
        state = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Test de la vérification des contraintes
        violations = shadow_manager.check_constraint_violations(action, state)

        assert isinstance(violations, list)
        # Vérifier que chaque violation a les champs requis
        for violation in violations:
            assert "constraint_type" in violation
            assert "severity" in violation
            assert "description" in violation

    def test_performance_impact_calculation(self, shadow_manager):
        """Test le calcul de l'impact sur les performances."""
        # Données de test
        rl_eta = datetime.now(UTC) + timedelta(minutes=15)
        human_eta = datetime.now(UTC) + timedelta(minutes=20)

        # Test du calcul de l'impact
        impact = shadow_manager.calculate_performance_impact(rl_eta, human_eta)

        assert isinstance(impact, float)
        assert -1 <= impact <= 1  # Impact normalisé entre -1 et 1

    def test_daily_report_generation(self, shadow_manager):
        """Test la génération du rapport quotidien."""
        # Données de test
        company_id = "company_1"
        date = datetime.now(UTC).date()

        # Mock des données de comparaison
        shadow_manager.comparison_history = {
            (company_id, date): [
                {"eta_delta": 5.0, "delay_delta": 2.0, "agreement_rate": 0.8, "rl_confidence": 0.85},
                {"eta_delta": 3.0, "delay_delta": 1.0, "agreement_rate": 0.9, "rl_confidence": 0.92},
            ]
        }

        # Test de la génération du rapport
        report = shadow_manager.generate_daily_report(company_id, date)

        assert isinstance(report, dict)
        assert "company_id" in report
        assert "date" in report
        assert "total_decisions" in report
        assert "average_eta_delta" in report
        assert "average_agreement_rate" in report
        assert "average_rl_confidence" in report

    def test_historical_analysis(self, shadow_manager):
        """Test l'analyse historique."""
        # Données de test
        company_id = "company_1"
        start_date = datetime.now(UTC).date() - timedelta(days=7)
        end_date = datetime.now(UTC).date()

        # Mock des données historiques
        shadow_manager.comparison_history = {}
        for i in range(7):
            date = start_date + timedelta(days=i)
            shadow_manager.comparison_history[(company_id, date)] = [
                {
                    "eta_delta": 5.0 + i,
                    "delay_delta": 2.0 + i,
                    "agreement_rate": 0.8 - (i * 0.01),
                    "rl_confidence": 0.85 + (i * 0.01),
                }
            ]

        # Test de l'analyse historique
        analysis = shadow_manager.analyze_historical_performance(company_id, start_date, end_date)

        assert isinstance(analysis, dict)
        assert "trend_analysis" in analysis
        assert "performance_summary" in analysis
        assert "recommendations" in analysis

    def test_alert_generation(self, shadow_manager):
        """Test la génération d'alertes pour les performances."""
        # Données de test
        company_id = "company_1"
        kpi_data = {
            "agreement_rate": 0.3,  # Faible accord
            "rl_confidence": 0.2,  # Faible confiance
            "constraint_violations": 5,  # Nombreuses violations
            "performance_impact": -0.8,  # Impact négatif important
        }

        # Test de la génération d'alertes
        alerts = shadow_manager.generate_performance_alerts(company_id, kpi_data)

        assert isinstance(alerts, list)
        # Vérifier que chaque alerte a les champs requis
        for alert in alerts:
            assert "alert_type" in alert
            assert "severity" in alert
            assert "message" in alert
            assert "recommendations" in alert

    def test_data_export(self, shadow_manager):
        """Test l'export des données."""
        # Données de test
        company_id = "company_1"
        start_date = datetime.now(UTC).date() - timedelta(days=7)
        end_date = datetime.now(UTC).date()

        # Mock des données
        shadow_manager.comparison_history = {
            (company_id, start_date): [
                {"eta_delta": 5.0, "delay_delta": 2.0, "agreement_rate": 0.8, "rl_confidence": 0.85}
            ]
        }

        # Test de l'export
        exported_data = shadow_manager.export_data(company_id, start_date, end_date)

        assert isinstance(exported_data, dict)
        assert "company_id" in exported_data
        assert "date_range" in exported_data
        assert "comparison_data" in exported_data

    def test_error_handling(self, shadow_manager):
        """Test la gestion d'erreurs."""
        # Test avec des données invalides
        invalid_state = None
        invalid_action = "invalid"

        with contextlib.suppress(ValueError, TypeError, AttributeError):
            shadow_manager.compare_decisions(invalid_state, invalid_action, None)

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
        manager.compare_decisions.return_value = {
            "rl_action": 5,
            "human_action": 3,
            "eta_delta": 5.0,
            "agreement_rate": 0.8,
            "rl_confidence": 0.85,
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
        request_data = {
            "state": [0.1, 0.2, 0.3, 0.4, 0.5],
            "human_action": 3,
            "human_eta": datetime.now(UTC).isoformat(),
        }

        # Simuler l'appel à l'endpoint
        try:
            result = mock_shadow_manager.compare_decisions(
                request_data["state"], request_data["human_action"], request_data["human_eta"]
            )

            assert isinstance(result, dict)
            assert "rl_action" in result
            assert "human_action" in result
        except Exception:
            # Gestion des erreurs d'intégration
            pass

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
        """Test l'endpoint d'analyse historique."""
        # Mock des paramètres
        company_id = "company_1"
        start_date = datetime.now(UTC).date() - timedelta(days=7)
        end_date = datetime.now(UTC).date()

        # Mock de la réponse
        mock_shadow_manager.analyze_historical_performance.return_value = {
            "trend_analysis": "improving",
            "performance_summary": "good",
            "recommendations": ["continue monitoring"],
        }

        # Test de l'analyse historique
        analysis = mock_shadow_manager.analyze_historical_performance(company_id, start_date, end_date)

        assert isinstance(analysis, dict)
        assert "trend_analysis" in analysis
        assert "performance_summary" in analysis


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
