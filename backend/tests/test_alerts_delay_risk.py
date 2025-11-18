#!/usr/bin/env python3
"""
Tests complets pour les alertes proactives et l'explicabilité RL.

Ce module teste:
- Service d'alertes proactives
- Routes REST pour alertes
- Socket.IO pour alertes temps réel
- Système de debounce anti-spam
- Explicabilité des décisions RL

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import gc
import json
import time
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from routes.proactive_alerts import _get_mock_booking_data, _get_mock_driver_data, proactive_alerts_bp
from services.proactive_alerts import ProactiveAlertsService


class TestProactiveAlertsService:
    """Tests pour le service d'alertes proactives."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.alerts_service = ProactiveAlertsService()

        # Données de test
        self.test_booking = {
            "id": "test_123",
            "pickup_lat": 46.2044,
            "pickup_lon": 6.1432,
            "pickup_time": (datetime.now(UTC) + timedelta(minutes=20)).isoformat(),
            "priority": 3,
            "is_outbound": True,
            "estimated_duration": 30,
        }

        self.test_driver = {
            "id": "driver_456",
            "lat": 46.2100,
            "lon": 6.1400,
            "current_bookings": 2,
            "load": 0,
            "type": "REGULAR",
            "available": True,
        }

    def test_service_initialization(self):
        """Test initialisation du service."""
        assert self.alerts_service is not None
        assert self.alerts_service.delay_risk_thresholds is not None
        assert self.alerts_service.debounce_minutes == 15
        assert isinstance(self.alerts_service.alert_history, dict)

    def test_check_delay_risk_basic(self):
        """Test analyse de risque de retard basique."""
        result = self.alerts_service.check_delay_risk(booking=self.test_booking, driver=self.test_driver)

        assert "delay_probability" in result
        assert "risk_level" in result
        assert "explanation" in result
        assert "metrics" in result
        assert "should_alert" in result

        assert 0.0 <= result["delay_probability"] <= 1.0
        assert result["risk_level"] in ["minimal", "low", "medium", "high", "unknown"]

    def test_check_delay_risk_high_risk_scenario(self):
        """Test scénario à haut risque."""
        # Booking avec peu de temps restant
        high_risk_booking = self.test_booking.copy()
        high_risk_booking["pickup_time"] = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()

        # Chauffeur loin
        far_driver = self.test_driver.copy()
        far_driver["lat"] = 46.3000  # Plus loin
        far_driver["lon"] = 6.2000

        result = self.alerts_service.check_delay_risk(booking=high_risk_booking, driver=far_driver)

        assert result["delay_probability"] > 0.5
        assert result["risk_level"] in ["medium", "high"]
        assert result["should_alert"] is True

    def test_check_delay_risk_low_risk_scenario(self):
        """Test scénario à faible risque."""
        # Booking avec beaucoup de temps
        low_risk_booking = self.test_booking.copy()
        low_risk_booking["pickup_time"] = (datetime.now(UTC) + timedelta(minutes=60)).isoformat()

        # Chauffeur proche
        close_driver = self.test_driver.copy()
        close_driver["lat"] = 46.2045  # Très proche
        close_driver["lon"] = 6.1433

        result = self.alerts_service.check_delay_risk(booking=low_risk_booking, driver=close_driver)

        assert result["delay_probability"] < 0.5
        assert result["risk_level"] in ["minimal", "low"]
        assert result["should_alert"] is False

    def test_heuristic_delay_probability(self):
        """Test calcul heuristique de probabilité."""
        # Test avec différents scénarios
        scenarios = [
            # Temps insuffisant
            {
                "booking": {**self.test_booking, "pickup_time": (datetime.now(UTC) + timedelta(minutes=5)).isoformat()},
                "driver": self.test_driver,
                "expected_min": 0.3,
            },
            # Distance importante
            {
                "booking": self.test_booking,
                "driver": {**self.test_driver, "lat": 46.3000, "lon": 6.2000},
                "expected_min": 0.2,
            },
            # Charge élevée
            {"booking": self.test_booking, "driver": {**self.test_driver, "current_bookings": 4}, "expected_min": 0.2},
        ]

        for scenario in scenarios:
            prob = self.alerts_service._heuristic_delay_probability(
                scenario["booking"], scenario["driver"], datetime.now(UTC)
            )
            assert prob >= scenario["expected_min"]

    def test_determine_risk_level(self):
        """Test détermination du niveau de risque."""
        test_cases = [(0.9, "high"), (0.7, "medium"), (0.4, "low"), (0.1, "minimal")]

        for probability, expected_level in test_cases:
            level = self.alerts_service._determine_risk_level(probability)
            assert level == expected_level

    def test_generate_explanation(self):
        """Test génération d'explication."""
        explanation = self.alerts_service._generate_explanation(
            booking=self.test_booking, driver=self.test_driver, probability=0.7, risk_level="medium"
        )

        assert "risk_level" in explanation
        assert "probability_percent" in explanation
        assert "primary_factors" in explanation
        assert "recommendations" in explanation
        assert "alternative_drivers" in explanation
        assert "business_impact" in explanation

        assert explanation["risk_level"] == "medium"
        assert explanation["probability_percent"] == 70.0

    def test_analyze_risk_factors(self):
        """Test analyse des facteurs de risque."""
        factors = self.alerts_service._analyze_risk_factors(self.test_booking, self.test_driver)

        assert isinstance(factors, list)
        for factor in factors:
            assert "factor" in factor
            assert "impact" in factor
            assert "description" in factor
            assert "value" in factor

    def test_generate_recommendations(self):
        """Test génération de recommandations."""
        recommendations = self.alerts_service._generate_recommendations(
            booking=self.test_booking, driver=self.test_driver, probability=0.8
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # Vérifier que les recommandations contiennent des emojis
        for rec in recommendations:
            assert any(emoji in rec for emoji in ["🚨", "⚠️", "📞", "🔄", "⏰"])

    def test_suggest_alternative_drivers(self):
        """Test suggestion de chauffeurs alternatifs."""
        alternatives = self.alerts_service._suggest_alternative_drivers(self.test_booking, self.test_driver)

        assert isinstance(alternatives, list)
        for alt in alternatives:
            assert "driver_id" in alt
            assert "estimated_distance" in alt
            assert "risk_reduction" in alt
            assert "reason" in alt

    def test_send_proactive_alert_success(self):
        """Test envoi d'alerte proactive réussi."""
        analysis_result = {
            "booking_id": "test_123",
            "driver_id": "driver_456",
            "delay_probability": 0.8,
            "risk_level": "high",
            "should_alert": True,
            "explanation": {"test": "explanation"},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        with patch.object(self.alerts_service, "_send_alert_notification", return_value=True):
            success = self.alerts_service.send_proactive_alert(
                analysis_result=analysis_result, company_id="test_company"
            )

            assert success is True
            assert "test_123" in self.alerts_service.alert_history

    def test_send_proactive_alert_debounce(self):
        """Test système de debounce."""
        analysis_result = {
            "booking_id": "test_123",
            "driver_id": "driver_456",
            "delay_probability": 0.8,
            "risk_level": "high",
            "should_alert": True,
            "explanation": {"test": "explanation"},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Premier envoi
        with patch.object(self.alerts_service, "_send_alert_notification", return_value=True):
            success1 = self.alerts_service.send_proactive_alert(
                analysis_result=analysis_result, company_id="test_company"
            )
            assert success1 is True

        # Deuxième envoi immédiat (doit être debounced)
        with patch.object(self.alerts_service, "_send_alert_notification", return_value=True) as mock_send:
            success2 = self.alerts_service.send_proactive_alert(
                analysis_result=analysis_result, company_id="test_company"
            )
            assert success2 is False
            mock_send.assert_not_called()

    def test_send_proactive_alert_force_send(self):
        """Test envoi forcé malgré debounce."""
        analysis_result = {
            "booking_id": "test_123",
            "driver_id": "driver_456",
            "delay_probability": 0.8,
            "risk_level": "high",
            "should_alert": True,
            "explanation": {"test": "explanation"},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Premier envoi
        with patch.object(self.alerts_service, "_send_alert_notification", return_value=True):
            self.alerts_service.send_proactive_alert(analysis_result=analysis_result, company_id="test_company")

        # Envoi forcé
        with patch.object(self.alerts_service, "_send_alert_notification", return_value=True) as mock_send:
            success = self.alerts_service.send_proactive_alert(
                analysis_result=analysis_result, company_id="test_company", force_send=True
            )
            assert success is True
            mock_send.assert_called_once()

    def test_get_explanation_for_decision(self):
        """Test génération d'explication pour décision RL."""
        rl_decision = {
            "q_values": {"action_1": 0.8, "action_2": 0.6, "action_3": 0.4},
            "confidence": 0.85,
            "reward_components": {"punctuality": 0.7, "distance": 0.3},
            "constraints_applied": ["time_window", "capacity"],
            "reward_profile": "PUNCTUALITY_FOCUSED",
            "action_masked": True,
        }

        explanation = self.alerts_service.get_explanation_for_decision(
            booking_id="test_123", driver_id="driver_456", rl_decision=rl_decision
        )

        assert explanation["decision_type"] == "rl_assignment"
        assert explanation["booking_id"] == "test_123"
        assert explanation["driver_id"] == "driver_456"
        assert "decision_factors" in explanation
        assert "alternative_options" in explanation
        assert "business_rules_applied" in explanation

    def test_clear_alert_history(self):
        """Test nettoyage de l'historique des alertes."""
        # Ajouter des alertes à l'historique
        self.alerts_service.alert_history["booking_1"] = {
            "last_alert_time": datetime.now(UTC),
            "last_risk_level": "medium",
            "total_alerts": 1,
        }
        self.alerts_service.alert_history["booking_2"] = {
            "last_alert_time": datetime.now(UTC),
            "last_risk_level": "high",
            "total_alerts": 2,
        }

        # Nettoyer un booking spécifique
        self.alerts_service.clear_alert_history("booking_1")
        assert "booking_1" not in self.alerts_service.alert_history
        assert "booking_2" in self.alerts_service.alert_history

        # Nettoyer tout
        self.alerts_service.clear_alert_history()
        assert len(self.alerts_service.alert_history) == 0

    def test_get_alert_statistics(self):
        """Test récupération des statistiques."""
        # Ajouter des alertes à l'historique
        self.alerts_service.alert_history["booking_1"] = {
            "last_alert_time": datetime.now(UTC),
            "last_risk_level": "medium",
            "total_alerts": 1,
        }
        self.alerts_service.alert_history["booking_2"] = {
            "last_alert_time": datetime.now(UTC),
            "last_risk_level": "high",
            "total_alerts": 2,
        }

        stats = self.alerts_service.get_alert_statistics()

        assert "total_alerts_sent" in stats
        assert "active_debounce_count" in stats
        assert "risk_level_distribution" in stats
        assert "debounce_minutes" in stats
        assert "delay_predictor_loaded" in stats

        assert stats["total_alerts_sent"] == 2
        assert stats["active_debounce_count"] == 2


class TestProactiveAlertsRoutes:
    """Tests pour les routes REST d'alertes proactives."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.app = MagicMock()
        self.client = MagicMock()

    def test_get_mock_booking_data(self):
        """Test récupération de données mock booking."""
        booking_data = _get_mock_booking_data("123")

        assert booking_data is not None
        assert booking_data["id"] == "123"
        assert "pickup_lat" in booking_data
        assert "pickup_lon" in booking_data
        assert "pickup_time" in booking_data

    def test_get_mock_driver_data(self):
        """Test récupération de données mock driver."""
        driver_data = _get_mock_driver_data("456")

        assert driver_data is not None
        assert driver_data["id"] == "456"
        assert "lat" in driver_data
        assert "lon" in driver_data
        assert "current_bookings" in driver_data

    def test_get_mock_data_not_found(self):
        """Test données mock non trouvées."""
        booking_data = _get_mock_booking_data("nonexistent")
        driver_data = _get_mock_driver_data("nonexistent")

        assert booking_data is None
        assert driver_data is None


class TestSocketIOIntegration:
    """Tests pour l'intégration Socket.IO."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.socketio = MagicMock()
        self.mock_client_id = "test_client_123"

    def test_broadcast_delay_alert(self):
        """Test diffusion d'alerte de retard."""
        from sockets.proactive_alerts import broadcast_delay_alert

        analysis_result = {
            "booking_id": "test_123",
            "driver_id": "driver_456",
            "delay_probability": 0.8,
            "risk_level": "high",
            "explanation": {"test": "explanation"},
        }

        success = broadcast_delay_alert(
            company_id="test_company", analysis_result=analysis_result, socketio=self.socketio
        )

        assert success is True
        self.socketio.emit.assert_called_once()

    def test_broadcast_rl_explanation(self):
        """Test diffusion d'explication RL."""
        from sockets.proactive_alerts import broadcast_rl_explanation

        explanation = {
            "booking_id": "test_123",
            "driver_id": "driver_456",
            "decision_factors": [],
            "alternative_options": [],
        }

        success = broadcast_rl_explanation(company_id="test_company", explanation=explanation, socketio=self.socketio)

        assert success is True
        self.socketio.emit.assert_called_once()

    def test_get_active_connections_stats(self):
        """Test statistiques des connexions actives."""
        from sockets.proactive_alerts import get_active_connections_stats

        stats = get_active_connections_stats()

        assert "total_companies" in stats
        assert "total_connections" in stats
        assert "companies" in stats
        assert "timestamp" in stats


class TestDebounceSystem:
    """Tests pour le système de debounce."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.alerts_service = ProactiveAlertsService()

    def test_debounce_timing(self):
        """Test timing du système de debounce."""
        booking_id = "test_booking_debounce"
        analysis_result = {
            "booking_id": booking_id,
            "driver_id": "driver_456",
            "delay_probability": 0.8,
            "risk_level": "high",
            "should_alert": True,
            "explanation": {"test": "explanation"},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Premier envoi
        with patch.object(self.alerts_service, "_send_alert_notification", return_value=True):
            success1 = self.alerts_service.send_proactive_alert(
                analysis_result=analysis_result, company_id="test_company"
            )
            assert success1 is True

        # Vérifier que l'alerte est dans l'historique
        assert booking_id in self.alerts_service.alert_history

        # Simuler le passage du temps (moins que le debounce)
        original_time = self.alerts_service.alert_history[booking_id]["last_alert_time"]
        self.alerts_service.alert_history[booking_id]["last_alert_time"] = original_time - timedelta(minutes=10)

        # Deuxième envoi (doit encore être debounced)
        with patch.object(self.alerts_service, "_send_alert_notification", return_value=True) as mock_send:
            success2 = self.alerts_service.send_proactive_alert(
                analysis_result=analysis_result, company_id="test_company"
            )
            assert success2 is False
            mock_send.assert_not_called()

        # Simuler le passage du temps (plus que le debounce)
        self.alerts_service.alert_history[booking_id]["last_alert_time"] = original_time - timedelta(minutes=20)

        # Troisième envoi (doit passer)
        with patch.object(self.alerts_service, "_send_alert_notification", return_value=True) as mock_send:
            success3 = self.alerts_service.send_proactive_alert(
                analysis_result=analysis_result, company_id="test_company"
            )
            assert success3 is True
            mock_send.assert_called_once()

    def test_debounce_different_bookings(self):
        """Test que le debounce ne s'applique qu'au même booking."""
        analysis_result_1 = {
            "booking_id": "booking_1",
            "driver_id": "driver_456",
            "delay_probability": 0.8,
            "risk_level": "high",
            "should_alert": True,
            "explanation": {"test": "explanation"},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        analysis_result_2 = {
            "booking_id": "booking_2",
            "driver_id": "driver_456",
            "delay_probability": 0.8,
            "risk_level": "high",
            "should_alert": True,
            "explanation": {"test": "explanation"},
            "timestamp": datetime.now(UTC).isoformat(),
        }

        with patch.object(self.alerts_service, "_send_alert_notification", return_value=True):
            # Premier booking
            success1 = self.alerts_service.send_proactive_alert(
                analysis_result=analysis_result_1, company_id="test_company"
            )
            assert success1 is True

            # Deuxième booking (doit passer car booking différent)
            success2 = self.alerts_service.send_proactive_alert(
                analysis_result=analysis_result_2, company_id="test_company"
            )
            assert success2 is True

            # Vérifier que les deux sont dans l'historique
            assert "booking_1" in self.alerts_service.alert_history
            assert "booking_2" in self.alerts_service.alert_history


class TestErrorHandling:
    """Tests pour la gestion d'erreurs."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.alerts_service = ProactiveAlertsService()

    def test_check_delay_risk_error_handling(self):
        """Test gestion d'erreur dans check_delay_risk."""
        # Booking invalide
        invalid_booking = None

        result = self.alerts_service.check_delay_risk(booking=invalid_booking or {}, driver={})

        assert result["delay_probability"] == 0.0
        assert result["risk_level"] == "unknown"
        assert "error" in result["explanation"]

    def test_send_alert_error_handling(self):
        """Test gestion d'erreur dans send_proactive_alert."""
        # Analysis result invalide
        invalid_analysis = None

        with pytest.raises((Exception, TypeError)):
            self.alerts_service.send_proactive_alert(analysis_result=invalid_analysis or {}, company_id="test_company")

    def test_explanation_error_handling(self):
        """Test gestion d'erreur dans get_explanation_for_decision."""
        # RL decision invalide
        invalid_rl_decision = None

        explanation = self.alerts_service.get_explanation_for_decision(
            booking_id="test_123", driver_id="driver_456", rl_decision=invalid_rl_decision or {}
        )

        assert "error" in explanation


class TestPerformanceMetrics:
    """Tests pour les métriques de performance."""

    def setup_method(self):
        """Setup pour chaque test."""
        self.alerts_service = ProactiveAlertsService()

    def test_analysis_performance(self):
        """Test performance de l'analyse de risque."""
        booking = {
            "id": "perf_test",
            "pickup_lat": 46.2044,
            "pickup_lon": 6.1432,
            "pickup_time": (datetime.now(UTC) + timedelta(minutes=30)).isoformat(),
            "priority": 3,
            "is_outbound": True,
            "estimated_duration": 30,
        }

        driver = {
            "id": "driver_perf",
            "lat": 46.2100,
            "lon": 6.1400,
            "current_bookings": 1,
            "load": 0,
            "type": "REGULAR",
            "available": True,
        }

        # Mesurer le temps d'exécution
        start_time = time.time()

        for _ in range(100):  # 100 analyses
            result = self.alerts_service.check_delay_risk(booking, driver)
            assert result is not None

        end_time = time.time()
        avg_time = (end_time - start_time) / 100

        # Vérifier que l'analyse est rapide (< 10ms en moyenne)
        assert avg_time < 0.01, f"Temps moyen trop élevé: {avg_time}"

    def test_memory_usage(self):
        """Test utilisation mémoire."""
        import sys

        # Mesurer la mémoire avant
        initial_objects = len(gc.get_objects()) if "gc" in globals() else 0

        # Créer beaucoup d'alertes
        for i in range(1000):
            analysis_result = {
                "booking_id": f"booking_{i}",
                "driver_id": f"driver_{i}",
                "delay_probability": 0.5,
                "risk_level": "medium",
                "should_alert": True,
                "explanation": {"test": f"explanation_{i}"},
                "timestamp": datetime.now(UTC).isoformat(),
            }

            self.alerts_service.send_proactive_alert(analysis_result=analysis_result, company_id="test_company")

        # Vérifier que la mémoire n'explose pas
        final_objects = len(gc.get_objects()) if "gc" in globals() else 0
        object_growth = final_objects - initial_objects

        # Le nombre d'objets ne doit pas croître de manière excessive
        assert object_growth < 10000, f"Trop d'objets créés: {object_growth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
