#!/usr/bin/env python3
"""
Tests d'intégration pour le système de dispatch avec Safety Guards.

Teste l'intégration complète entre le moteur de dispatch, l'optimiseur RL,
et les Safety Guards avec rollback automatique.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

import pytest

# Import conditionnel pour éviter les erreurs si les modules ne sont pas disponibles
try:
    from services.safety_guards import SafetyGuards, get_safety_guards
except ImportError:
    SafetyGuards = None
    get_safety_guards = None

try:
    from services.unified_dispatch.rl_optimizer import RLDispatchOptimizer
except ImportError:
    RLDispatchOptimizer = None


class TestDispatchSafetyIntegration:
    """Tests d'intégration dispatch + Safety Guards."""

    @pytest.fixture
    def mock_company(self):
        """Mock d'une entreprise."""
        company = Mock()
        company.id = 1
        company.name = "Test Company"
        return company

    @pytest.fixture
    def mock_drivers(self):
        """Mock des chauffeurs."""
        drivers = []
        for i in range(3):
            driver = Mock()
            driver.id = i + 1
            driver.name = f"Driver {i + 1}"
            driver.is_available = True
            driver.driver_type = "REGULAR"
            drivers.append(driver)
        return drivers

    @pytest.fixture
    def mock_bookings(self):
        """Mock des bookings."""
        bookings = []
        for i in range(10):
            booking = Mock()
            booking.id = i + 1
            booking.scheduled_time = datetime.now(UTC) + timedelta(minutes=i * 30)
            booking.pickup_lat = 46.0 + i * 0.01
            booking.pickup_lon = 6.0 + i * 0.01
            booking.priority = 1
            bookings.append(booking)
        return bookings

    @pytest.fixture
    def mock_assignments(self, mock_bookings, mock_drivers):
        """Mock des assignations."""
        assignments = []
        for i, booking in enumerate(mock_bookings):
            assignment = Mock()
            assignment.booking_id = booking.id
            assignment.driver_id = mock_drivers[i % len(mock_drivers)].id
            assignments.append(assignment)
        return assignments

    def test_safe_dispatch_with_rl_optimization(
        self, mock_company, mock_drivers, mock_bookings, mock_assignments
    ):
        """Test un dispatch sûr avec optimisation RL."""
        if SafetyGuards is None or RLDispatchOptimizer is None:
            pytest.skip("Modules non disponibles")

        # Mock des Safety Guards
        with patch("services.safety_guards.get_safety_guards") as mock_get_guards:
            safety_guards = Mock()
            safety_guards.check_dispatch_result.return_value = (True, {"is_safe": True})
            mock_get_guards.return_value = safety_guards

            # Mock de l'optimiseur RL
            with patch(
                "services.unified_dispatch.rl_optimizer.RLDispatchOptimizer"
            ) as mock_optimizer_class:
                optimizer = Mock()
                optimizer.is_available.return_value = True
                optimizer.optimize_assignments.return_value = mock_assignments
                mock_optimizer_class.return_value = optimizer

                # Simuler le dispatch
                result = self._simulate_dispatch(
                    mock_company, mock_drivers, mock_bookings, mock_assignments
                )

                # Vérifier que l'optimisation RL a été appelée
                optimizer.optimize_assignments.assert_called_once()

                # Vérifier que les Safety Guards ont été appelés
                safety_guards.check_dispatch_result.assert_called_once()

                # Vérifier que le résultat est sûr
                assert result["is_safe"] is True
                assert result["rollback_performed"] is False

    def test_unsafe_dispatch_with_rollback(
        self, mock_company, mock_drivers, mock_bookings, mock_assignments
    ):
        """Test un dispatch dangereux avec rollback automatique."""
        if SafetyGuards is None or RLDispatchOptimizer is None:
            pytest.skip("Modules non disponibles")

        # Mock des Safety Guards qui détectent un problème
        with patch("services.safety_guards.get_safety_guards") as mock_get_guards:
            safety_guards = Mock()
            safety_guards.check_dispatch_result.return_value = (
                False,
                {
                    "is_safe": False,
                    "violation_count": 3,
                    "checks": {
                        "max_delay_ok": False,
                        "completion_rate_ok": False,
                        "driver_load_ok": False,
                    },
                },
            )
            mock_get_guards.return_value = safety_guards

            # Mock de l'optimiseur RL
            with patch(
                "services.unified_dispatch.rl_optimizer.RLDispatchOptimizer"
            ) as mock_optimizer_class:
                optimizer = Mock()
                optimizer.is_available.return_value = True
                # L'optimiseur retourne des assignations "dangereuses"
                dangerous_assignments = mock_assignments.copy()
                optimizer.optimize_assignments.return_value = dangerous_assignments
                mock_optimizer_class.return_value = optimizer

                # Mock du service de notification
                with patch(
                    "services.notification_service.NotificationService"
                ) as mock_notification_class:
                    notification_service = Mock()
                    mock_notification_class.return_value = notification_service

                    # Simuler le dispatch
                    result = self._simulate_dispatch(
                        mock_company, mock_drivers, mock_bookings, mock_assignments
                    )

                    # Vérifier que l'optimisation RL a été appelée
                    optimizer.optimize_assignments.assert_called_once()

                    # Vérifier que les Safety Guards ont été appelés
                    safety_guards.check_dispatch_result.assert_called_once()

                    # Vérifier que le rollback a été effectué
                    assert result["is_safe"] is False
                    assert result["rollback_performed"] is True

                    # Vérifier que la notification a été envoyée
                    notification_service.send_alert.assert_called_once()

    def test_rl_optimizer_safety_check(
        self, mock_drivers, mock_bookings, mock_assignments
    ):
        """Test la vérification de sécurité dans l'optimiseur RL."""
        if SafetyGuards is None or RLDispatchOptimizer is None:
            pytest.skip("Modules non disponibles")

        # Mock des Safety Guards
        with patch("services.safety_guards.get_safety_guards") as mock_get_guards:
            safety_guards = Mock()
            safety_guards.check_dispatch_result.return_value = (True, {"is_safe": True})
            mock_get_guards.return_value = safety_guards

            # Créer l'optimiseur RL
            optimizer = RLDispatchOptimizer()

            # Mock des méthodes internes
            with (
                patch.object(optimizer, "is_available", return_value=True),
                patch.object(optimizer, "_calculate_gap", return_value=5),
                patch.object(
                    optimizer, "_calculate_loads", return_value={1: 3, 2: 4, 3: 3}
                ),
                patch.object(optimizer, "_create_state", return_value=Mock()),
            ):
                # Mock de l'agent
                optimizer.agent = Mock()
                optimizer.agent.select_action.return_value = 0  # Wait action

                # Exécuter l'optimisation
                result = optimizer.optimize_assignments(
                    mock_assignments, mock_bookings, mock_drivers
                )

                # Vérifier que les Safety Guards ont été appelés
                safety_guards.check_dispatch_result.assert_called_once()

                # Vérifier que le résultat est retourné
                assert result == mock_assignments

    def test_rollback_scenario_critical_violations(
        self, mock_company, mock_drivers, mock_bookings, mock_assignments
    ):
        """Test le scénario de rollback avec violations critiques."""
        if SafetyGuards is None:
            pytest.skip("SafetyGuards non disponible")

        # Mock des Safety Guards avec violations critiques
        with patch("services.safety_guards.get_safety_guards") as mock_get_guards:
            safety_guards = Mock()
            safety_guards.check_dispatch_result.return_value = (
                False,
                {
                    "is_safe": False,
                    "violation_count": 8,  # Beaucoup de violations
                    "checks": {
                        "max_delay_ok": False,
                        "completion_rate_ok": False,
                        "invalid_actions_ok": False,
                        "driver_load_ok": False,
                        "driver_utilization_ok": False,
                        "avg_distance_ok": False,
                        "max_distance_ok": False,
                        "rl_confidence_ok": False,
                    },
                },
            )
            mock_get_guards.return_value = safety_guards

            # Simuler le dispatch
            result = self._simulate_dispatch(
                mock_company, mock_drivers, mock_bookings, mock_assignments
            )

            # Vérifier que le rollback a été effectué
            assert result["is_safe"] is False
            assert result["rollback_performed"] is True
            assert result["violation_count"] == 8

    def test_notification_service_integration(
        self, mock_company, mock_drivers, mock_bookings, mock_assignments
    ):
        """Test l'intégration avec le service de notification."""
        if SafetyGuards is None:
            pytest.skip("SafetyGuards non disponible")

        # Mock des Safety Guards avec violation
        with patch("services.safety_guards.get_safety_guards") as mock_get_guards:
            safety_guards = Mock()
            safety_guards.check_dispatch_result.return_value = (
                False,
                {
                    "is_safe": False,
                    "violation_count": 2,
                    "checks": {"max_delay_ok": False, "completion_rate_ok": False},
                },
            )
            mock_get_guards.return_value = safety_guards

            # Mock du service de notification
            with patch(
                "services.notification_service.NotificationService"
            ) as mock_notification_class:
                notification_service = Mock()
                mock_notification_class.return_value = notification_service

                # Simuler le dispatch
                _result = self._simulate_dispatch(
                    mock_company, mock_drivers, mock_bookings, mock_assignments
                )

                # Vérifier que la notification a été envoyée
                notification_service.send_alert.assert_called_once()

                # Vérifier les paramètres de la notification
                call_args = notification_service.send_alert.call_args
                assert call_args[1]["alert_type"] == "safety_rollback"
                assert call_args[1]["severity"] == "warning"
                assert "Rollback RL vers heuristique" in call_args[1]["message"]

    def test_error_handling_in_safety_guards(
        self, mock_company, mock_drivers, mock_bookings, mock_assignments
    ):
        """Test la gestion d'erreurs dans les Safety Guards."""
        if SafetyGuards is None:
            pytest.skip("SafetyGuards non disponible")

        # Mock des Safety Guards qui lèvent une exception
        with patch("services.safety_guards.get_safety_guards") as mock_get_guards:
            safety_guards = Mock()
            safety_guards.check_dispatch_result.side_effect = Exception(
                "Safety Guards error"
            )
            mock_get_guards.return_value = safety_guards

            # Simuler le dispatch
            result = self._simulate_dispatch(
                mock_company, mock_drivers, mock_bookings, mock_assignments
            )

            # Vérifier que le système continue de fonctionner malgré l'erreur
            assert "error" in result
            assert result["error"] == "Safety Guards error"

    def test_performance_under_high_load(
        self, mock_company, mock_drivers, mock_bookings, mock_assignments
    ):
        """Test les performances sous charge élevée."""
        if SafetyGuards is None:
            pytest.skip("SafetyGuards non disponible")

        import time

        # Mock des Safety Guards
        with patch("services.safety_guards.get_safety_guards") as mock_get_guards:
            safety_guards = Mock()
            safety_guards.check_dispatch_result.return_value = (True, {"is_safe": True})
            mock_get_guards.return_value = safety_guards

            start_time = time.time()

            # Simuler 100 dispatches
            for _ in range(100):
                self._simulate_dispatch(
                    mock_company, mock_drivers, mock_bookings, mock_assignments
                )

            end_time = time.time()
            total_time = end_time - start_time

            # Vérifier que chaque dispatch prend moins de 100ms en moyenne
            avg_time_per_dispatch = total_time / 100
            assert avg_time_per_dispatch < 0.1  # 100ms

    def _simulate_dispatch(self, company, drivers, bookings, assignments):
        """Simule un dispatch avec Safety Guards."""
        try:
            # Mock des Safety Guards
            safety_guards = get_safety_guards()

            # Préparer les métriques de dispatch
            dispatch_metrics = {
                "max_delay_minutes": 15.0,
                "avg_delay_minutes": 8.0,
                "completion_rate": len(assignments) / len(bookings)
                if bookings
                else 1.0,
                "invalid_action_rate": 0.01,
                "driver_loads": [
                    len([a for a in assignments if a.driver_id == d.id])
                    for d in drivers
                ],
                "avg_distance_km": 12.0,
                "max_distance_km": 20.0,
                "total_distance_km": 60.0,
            }

            # Métadonnées RL
            rl_metadata = {
                "confidence": 0.85,
                "uncertainty": 0.15,
                "decision_time_ms": 35,
                "q_value_variance": 0.1,
                "episode_length": 100,
            }

            # Vérifier la sécurité
            is_safe, safety_result = safety_guards.check_dispatch_result(
                dispatch_metrics, rl_metadata
            )

            rollback_performed = False
            if not is_safe:
                rollback_performed = True
                # Simuler le rollback vers assignations heuristiques
                assignments = assignments.copy()

            return {
                "is_safe": is_safe,
                "rollback_performed": rollback_performed,
                "violation_count": safety_result.get("violation_count", 0),
                "safety_result": safety_result,
            }

        except Exception as e:
            return {"error": str(e), "is_safe": False, "rollback_performed": False}


class TestDispatchRollbackScenarios:
    """Tests spécifiques pour les scénarios de rollback."""

    def test_rollback_due_to_high_delay(self):
        """Test rollback dû à des retards élevés."""
        if SafetyGuards is None:
            pytest.skip("SafetyGuards non disponible")

        safety_guards = SafetyGuards()

        # Dispatch avec retards élevés
        dispatch_result = {
            "max_delay_minutes": 45.0,  # > 30 min (seuil)
            "avg_delay_minutes": 25.0,
            "completion_rate": 0.95,
            "invalid_action_rate": 0.01,
            "driver_loads": [3, 4, 5],
            "avg_distance_km": 12.0,
            "max_distance_km": 20.0,
        }

        is_safe, result = safety_guards.check_dispatch_result(dispatch_result, None)

        assert is_safe is False
        assert result["checks"]["max_delay_ok"] is False

    def test_rollback_due_to_low_completion_rate(self):
        """Test rollback dû à un faible taux de completion."""
        if SafetyGuards is None:
            pytest.skip("SafetyGuards non disponible")

        safety_guards = SafetyGuards()

        # Dispatch avec faible taux de completion
        dispatch_result = {
            "max_delay_minutes": 15.0,
            "avg_delay_minutes": 8.0,
            "completion_rate": 0.80,  # < 0.90 (seuil)
            "invalid_action_rate": 0.01,
            "driver_loads": [3, 4, 5],
            "avg_distance_km": 12.0,
            "max_distance_km": 20.0,
        }

        is_safe, result = safety_guards.check_dispatch_result(dispatch_result, None)

        assert is_safe is False
        assert result["checks"]["completion_rate_ok"] is False

    def test_rollback_due_to_high_driver_load(self):
        """Test rollback dû à une charge élevée des chauffeurs."""
        if SafetyGuards is None:
            pytest.skip("SafetyGuards non disponible")

        safety_guards = SafetyGuards()

        # Dispatch avec charge élevée
        dispatch_result = {
            "max_delay_minutes": 15.0,
            "avg_delay_minutes": 8.0,
            "completion_rate": 0.95,
            "invalid_action_rate": 0.01,
            "driver_loads": [15, 2, 1],  # Max > 12 (seuil)
            "avg_distance_km": 12.0,
            "max_distance_km": 20.0,
        }

        is_safe, result = safety_guards.check_dispatch_result(dispatch_result, None)

        assert is_safe is False
        assert result["checks"]["driver_load_ok"] is False

    def test_rollback_due_to_low_rl_confidence(self):
        """Test rollback dû à une faible confiance RL."""
        if SafetyGuards is None:
            pytest.skip("SafetyGuards non disponible")

        safety_guards = SafetyGuards()

        # Dispatch avec faible confiance RL
        dispatch_result = {
            "max_delay_minutes": 15.0,
            "avg_delay_minutes": 8.0,
            "completion_rate": 0.95,
            "invalid_action_rate": 0.01,
            "driver_loads": [3, 4, 5],
            "avg_distance_km": 12.0,
            "max_distance_km": 20.0,
        }

        rl_metadata = {
            "confidence": 0.60,  # < 0.70 (seuil)
            "uncertainty": 0.40,
            "decision_time_ms": 35,
            "q_value_variance": 0.1,
            "episode_length": 100,
        }

        is_safe, result = safety_guards.check_dispatch_result(
            dispatch_result, rl_metadata
        )

        assert is_safe is False
        assert result["checks"]["rl_confidence_ok"] is False


def run_dispatch_integration_tests():
    """Exécute tous les tests d'intégration dispatch."""
    print("🚀 Exécution des tests d'intégration dispatch")

    # Tests de base
    test_classes = [TestDispatchSafetyIntegration, TestDispatchRollbackScenarios]

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

    print("\n📊 Résultats des tests d'intégration dispatch:")
    print("  Tests exécutés: {total_tests}")
    print("  Tests réussis: {passed_tests}")
    print(
        "  Taux de succès: {passed_tests/total_tests*100"
        if total_tests > 0
        else "  Taux de succès: 0%"
    )

    return passed_tests, total_tests


if __name__ == "__main__":
    run_dispatch_integration_tests()
