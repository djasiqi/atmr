"""Tests pour les méthodes non couvertes de shadow_mode_manager.py."""

import json
import math
from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from services.rl.shadow_mode_manager import ShadowModeManager


class TestShadowModeManagerUncovered:
    """Tests pour les méthodes non couvertes de ShadowModeManager."""

    def test_extract_decision_reasons_basic(self):
        """Test extraction raisons décision basique."""
        manager = ShadowModeManager()

        rl_decision = {
            "eta_minutes": 15,
            "distance_km": 5.0,
            "driver_load": 2,
            "respects_time_window": True,
            "driver_available": True,
            "driver_id": 1,
        }

        context = {"avg_eta": 18, "avg_distance": 7.0, "avg_load": 3, "driver_performance": {1: {"rating": 4.5}}}

        reasons = manager._extract_decision_reasons(rl_decision, context)

        assert isinstance(reasons, list)
        assert "ETA inférieur à la moyenne" in reasons
        assert "Distance optimisée" in reasons
        assert "Charge chauffeur équilibrée" in reasons
        assert "Respecte la fenêtre horaire" in reasons
        assert "Chauffeur disponible" in reasons
        assert "Chauffeur bien noté" in reasons

    def test_extract_decision_reasons_no_context(self):
        """Test extraction raisons décision sans contexte."""
        manager = ShadowModeManager()

        rl_decision = {
            "eta_minutes": 15,
            "distance_km": 5.0,
            "driver_load": 2,
            "respects_time_window": True,
            "driver_available": True,
            "driver_id": 1,
        }

        context = {}

        reasons = manager._extract_decision_reasons(rl_decision, context)

        assert isinstance(reasons, list)
        assert "Respecte la fenêtre horaire" in reasons
        assert "Chauffeur disponible" in reasons

    def test_extract_decision_reasons_with_violations(self):
        """Test extraction raisons décision avec violations."""
        manager = ShadowModeManager()

        rl_decision = {
            "eta_minutes": 25,
            "distance_km": 10.0,
            "driver_load": 5,
            "respects_time_window": False,
            "driver_available": False,
            "driver_id": 2,
        }

        context = {"avg_eta": 18, "avg_distance": 7.0, "avg_load": 3, "driver_performance": {2: {"rating": 3.5}}}

        reasons = manager._extract_decision_reasons(rl_decision, context)

        assert isinstance(reasons, list)
        # Ne devrait pas contenir les raisons positives
        assert "ETA inférieur à la moyenne" not in reasons
        assert "Distance optimisée" not in reasons
        assert "Charge chauffeur équilibrée" not in reasons
        assert "Respecte la fenêtre horaire" not in reasons
        assert "Chauffeur disponible" not in reasons
        assert "Chauffeur bien noté" not in reasons

    def test_check_constraint_violations_basic(self):
        """Test vérification violations contraintes basique."""
        manager = ShadowModeManager()

        rl_decision = {
            "respects_time_window": True,
            "driver_available": True,
            "passenger_count": 2,
            "in_service_area": True,
        }

        context = {"vehicle_capacity": 4}

        violations = manager._check_constraint_violations(rl_decision, context)

        assert isinstance(violations, list)
        assert len(violations) == 0  # Aucune violation

    def test_check_constraint_violations_time_window(self):
        """Test vérification violations fenêtre horaire."""
        manager = ShadowModeManager()

        rl_decision = {
            "respects_time_window": False,
            "driver_available": True,
            "passenger_count": 2,
            "in_service_area": True,
        }

        context = {"vehicle_capacity": 4}

        violations = manager._check_constraint_violations(rl_decision, context)

        assert isinstance(violations, list)
        assert "Fenêtre horaire non respectée" in violations

    def test_check_constraint_violations_driver_availability(self):
        """Test vérification violations disponibilité conducteur."""
        manager = ShadowModeManager()

        rl_decision = {
            "respects_time_window": True,
            "driver_available": False,
            "passenger_count": 2,
            "in_service_area": True,
        }

        context = {"vehicle_capacity": 4}

        violations = manager._check_constraint_violations(rl_decision, context)

        assert isinstance(violations, list)
        assert "Chauffeur non disponible" in violations

    def test_check_constraint_violations_capacity(self):
        """Test vérification violations capacité."""
        manager = ShadowModeManager()

        rl_decision = {
            "respects_time_window": True,
            "driver_available": True,
            "passenger_count": 6,
            "in_service_area": True,
        }

        context = {"vehicle_capacity": 4}

        violations = manager._check_constraint_violations(rl_decision, context)

        assert isinstance(violations, list)
        assert "Capacité véhicule dépassée" in violations

    def test_check_constraint_violations_service_area(self):
        """Test vérification violations zone de service."""
        manager = ShadowModeManager()

        rl_decision = {
            "respects_time_window": True,
            "driver_available": True,
            "passenger_count": 2,
            "in_service_area": False,
        }

        context = {"vehicle_capacity": 4}

        violations = manager._check_constraint_violations(rl_decision, context)

        assert isinstance(violations, list)
        assert "Hors zone de service" in violations

    def test_check_constraint_violations_multiple(self):
        """Test vérification violations multiples."""
        manager = ShadowModeManager()

        rl_decision = {
            "respects_time_window": False,
            "driver_available": False,
            "passenger_count": 6,
            "in_service_area": False,
        }

        context = {"vehicle_capacity": 4}

        violations = manager._check_constraint_violations(rl_decision, context)

        assert isinstance(violations, list)
        assert len(violations) == 4
        assert "Fenêtre horaire non respectée" in violations
        assert "Chauffeur non disponible" in violations
        assert "Capacité véhicule dépassée" in violations
        assert "Hors zone de service" in violations

    def test_check_constraint_violations_no_context(self):
        """Test vérification violations sans contexte."""
        manager = ShadowModeManager()

        rl_decision = {
            "respects_time_window": True,
            "driver_available": True,
            "passenger_count": 2,
            "in_service_area": True,
        }

        context = {}

        violations = manager._check_constraint_violations(rl_decision, context)

        assert isinstance(violations, list)
        assert len(violations) == 0  # Aucune violation

    def test_check_constraint_violations_missing_keys(self):
        """Test vérification violations avec clés manquantes."""
        manager = ShadowModeManager()

        rl_decision = {}

        context = {}

        violations = manager._check_constraint_violations(rl_decision, context)

        assert isinstance(violations, list)
        assert len(violations) == 0  # Aucune violation

    def test_calculate_performance_impact_basic(self):
        """Test calcul impact performance basique."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20, "delay_minutes": 5}
        rl_decision = {"eta_minutes": 15, "delay_minutes": 2}
        context = {"avg_eta": 18, "avg_delay": 3, "total_bookings": 100}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_no_context(self):
        """Test calcul impact performance sans contexte."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20, "delay_minutes": 5}
        rl_decision = {"eta_minutes": 15, "delay_minutes": 2}
        context = {}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_missing_values(self):
        """Test calcul impact performance avec valeurs manquantes."""
        manager = ShadowModeManager()

        human_decision = {}
        rl_decision = {}
        context = {}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_negative_values(self):
        """Test calcul impact performance avec valeurs négatives."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": -5, "delay_minutes": -2}
        rl_decision = {"eta_minutes": -3, "delay_minutes": -1}
        context = {"avg_eta": -4, "avg_delay": -1.5, "total_bookings": 100}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_large_values(self):
        """Test calcul impact performance avec valeurs importantes."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 1000, "delay_minutes": 100}
        rl_decision = {"eta_minutes": 500, "delay_minutes": 50}
        context = {"avg_eta": 750, "avg_delay": 75, "total_bookings": 1000}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_zero_values(self):
        """Test calcul impact performance avec valeurs zéro."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 0, "delay_minutes": 0}
        rl_decision = {"eta_minutes": 0, "delay_minutes": 0}
        context = {"avg_eta": 0, "avg_delay": 0, "total_bookings": 0}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_float_values(self):
        """Test calcul impact performance avec valeurs float."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20.5, "delay_minutes": 2.7}
        rl_decision = {"eta_minutes": 15.3, "delay_minutes": 1.8}
        context = {"avg_eta": 17.9, "avg_delay": 2.25, "total_bookings": 150.5}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_string_values(self):
        """Test calcul impact performance avec valeurs string."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": "20", "delay_minutes": "5"}
        rl_decision = {"eta_minutes": "15", "delay_minutes": "2"}
        context = {"avg_eta": "18", "avg_delay": "3", "total_bookings": "100"}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_mixed_types(self):
        """Test calcul impact performance avec types mixtes."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20, "delay_minutes": "5"}
        rl_decision = {"eta_minutes": "15", "delay_minutes": 2}
        context = {"avg_eta": 18, "avg_delay": "3", "total_bookings": 100}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_none_values(self):
        """Test calcul impact performance avec valeurs None."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": None, "delay_minutes": None}
        rl_decision = {"eta_minutes": None, "delay_minutes": None}
        context = {"avg_eta": None, "avg_delay": None, "total_bookings": None}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_invalid_types(self):
        """Test calcul impact performance avec types invalides."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": [20], "delay_minutes": {"value": 5}}
        rl_decision = {"eta_minutes": [15], "delay_minutes": {"value": 2}}
        context = {"avg_eta": [18], "avg_delay": {"value": 3}, "total_bookings": [100]}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_boolean_values(self):
        """Test calcul impact performance avec valeurs booléennes."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": True, "delay_minutes": False}
        rl_decision = {"eta_minutes": False, "delay_minutes": True}
        context = {"avg_eta": True, "avg_delay": False, "total_bookings": True}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_empty_strings(self):
        """Test calcul impact performance avec chaînes vides."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": "", "delay_minutes": ""}
        rl_decision = {"eta_minutes": "", "delay_minutes": ""}
        context = {"avg_eta": "", "avg_delay": "", "total_bookings": ""}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_whitespace_strings(self):
        """Test calcul impact performance avec chaînes avec espaces."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": " 20 ", "delay_minutes": " 5 "}
        rl_decision = {"eta_minutes": " 15 ", "delay_minutes": " 2 "}
        context = {"avg_eta": " 18 ", "avg_delay": " 3 ", "total_bookings": " 100 "}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_scientific_notation(self):
        """Test calcul impact performance avec notation scientifique."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 1e2, "delay_minutes": 1e1}
        rl_decision = {"eta_minutes": 5e1, "delay_minutes": 5e0}
        context = {"avg_eta": 7.5e1, "avg_delay": 7.5e0, "total_bookings": 1e2}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_infinity_values(self):
        """Test calcul impact performance avec valeurs infinies."""
        import math

        manager = ShadowModeManager()

        human_decision = {"eta_minutes": math.inf, "delay_minutes": -math.inf}
        rl_decision = {"eta_minutes": math.inf, "delay_minutes": -math.inf}
        context = {"avg_eta": math.inf, "avg_delay": -math.inf, "total_bookings": math.inf}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_nan_values(self):
        """Test calcul impact performance avec valeurs NaN."""

        manager = ShadowModeManager()

        human_decision = {"eta_minutes": math.nan, "delay_minutes": math.nan}
        rl_decision = {"eta_minutes": math.nan, "delay_minutes": math.nan}
        context = {"avg_eta": math.nan, "avg_delay": math.nan, "total_bookings": math.nan}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_extreme_values(self):
        """Test calcul impact performance avec valeurs extrêmes."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 999999, "delay_minutes": 0.0001}
        rl_decision = {"eta_minutes": 0.0001, "delay_minutes": 999999}
        context = {"avg_eta": 499999.5, "avg_delay": 499999.5, "total_bookings": 999999}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_very_small_values(self):
        """Test calcul impact performance avec valeurs très petites."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 1e-10, "delay_minutes": 1e-15}
        rl_decision = {"eta_minutes": 1e-11, "delay_minutes": 1e-16}
        context = {"avg_eta": 5.5e-11, "avg_delay": 5.5e-16, "total_bookings": 1e-10}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_very_large_values(self):
        """Test calcul impact performance avec valeurs très importantes."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 1e10, "delay_minutes": 1e15}
        rl_decision = {"eta_minutes": 1e9, "delay_minutes": 1e14}
        context = {"avg_eta": 5.5e9, "avg_delay": 5.5e14, "total_bookings": 1e10}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_mixed_precision(self):
        """Test calcul impact performance avec précision mixte."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20.123456789, "delay_minutes": 5.987654321}
        rl_decision = {"eta_minutes": 15.111111111, "delay_minutes": 2.999999999}
        context = {"avg_eta": 17.61728395, "avg_delay": 4.49382716, "total_bookings": 150.555555555}

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)

    def test_calculate_performance_impact_with_complex_scenario(self):
        """Test calcul impact performance avec scénario complexe."""
        manager = ShadowModeManager()

        human_decision = {
            "driver_id": 1,
            "eta_minutes": 20.5,
            "delay_minutes": 3.7,
            "confidence": 0.85,
            "reason": "Driver is closest",
            "traffic_factor": 1.2,
            "weather_factor": 1.1,
        }

        rl_decision = {
            "driver_id": 2,
            "eta_minutes": 17.3,
            "delay_minutes": 2.1,
            "confidence": 0.92,
            "reason": "Better route optimization",
            "traffic_factor": 1.0,
            "weather_factor": 1.0,
        }

        context = {
            "booking_id": 1,
            "pickup_time": datetime.now(),
            "distance_km": 5.0,
            "traffic_level": "high",
            "weather": "rain",
            "time_of_day": "rush_hour",
            "driver_experience": "expert",
            "vehicle_type": "sedan",
            "avg_eta": 18.9,
            "avg_delay": 2.9,
            "total_bookings": 125.7,
        }

        impact = manager._calculate_performance_impact(human_decision, rl_decision, context)

        assert isinstance(impact, float)
