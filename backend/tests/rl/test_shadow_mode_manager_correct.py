"""Tests corrects pour shadow_mode_manager.py."""

import json
from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from services.rl.shadow_mode_manager import ShadowModeManager


class TestShadowModeManagerCorrect:
    """Tests corrects pour ShadowModeManager."""

    def test_init_basic(self):
        """Test initialisation basique."""
        manager = ShadowModeManager()

        assert manager.data_dir is not None
        assert isinstance(manager.kpi_metrics, dict)
        assert isinstance(manager.decision_metadata, dict)
        assert manager.logger is not None

    def test_init_with_custom_data_dir(self):
        """Test initialisation avec répertoire personnalisé."""
        manager = ShadowModeManager(data_dir="custom/shadow/data")

        assert str(manager.data_dir) == "custom/shadow/data"

    def test_setup_logging(self):
        """Test configuration logging."""
        manager = ShadowModeManager()

        # Vérifier que le logger est configuré
        assert manager.logger is not None
        assert manager.logger.name == "services.rl.shadow_mode_manager"

    def test_log_decision_comparison_basic(self):
        """Test logging comparaison décisions basique."""
        manager = ShadowModeManager()

        human_decision = {
            "driver_id": 1,
            "eta_minutes": 15,
            "delay_minutes": 0
        }

        rl_decision = {
            "driver_id": 2,
            "eta_minutes": 12,
            "delay_minutes": 0
        }

        context = {
            "booking_id": 1,
            "pickup_time": datetime.now(),
            "distance_km": 5.0
        }

        kpis = manager.log_decision_comparison(
            company_id="1",
            booking_id="1",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis
        assert len(manager.decision_metadata["timestamp"]) == 1
        assert manager.decision_metadata["company_id"][0] == "1"
        assert manager.decision_metadata["booking_id"][0] == "1"

    def test_log_decision_comparison_with_none_values(self):
        """Test logging avec valeurs None."""
        manager = ShadowModeManager()

        human_decision = {
            "driver_id": 1,
            "eta_minutes": None,
            "delay_minutes": None
        }

        rl_decision = {
            "driver_id": 2,
            "eta_minutes": None,
            "delay_minutes": None
        }

        context = {
            "booking_id": 1,
            "pickup_time": datetime.now(),
            "distance_km": 5.0
        }

        kpis = manager.log_decision_comparison(
            company_id="1",
            booking_id="1",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == 0  # None - None = 0
        assert kpis["delay_delta"] == 0  # None - None = 0

    def test_calculate_kpis_basic(self):
        """Test calcul KPIs basique."""
        manager = ShadowModeManager()

        human_decision = {
            "driver_id": 1,
            "eta_minutes": 15,
            "delay_minutes": 0
        }

        rl_decision = {
            "driver_id": 2,
            "eta_minutes": 12,
            "delay_minutes": 0
        }

        context = {
            "booking_id": 1,
            "pickup_time": datetime.now(),
            "distance_km": 5.0
        }

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis
        assert "second_best_driver" in kpis
        assert "rl_confidence" in kpis
        assert "human_confidence" in kpis
        assert "decision_reasons" in kpis
        assert "constraint_violations" in kpis
        assert "performance_impact" in kpis

    def test_calculate_kpis_eta_delta(self):
        """Test calcul delta ETA."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20}
        rl_decision = {"eta_minutes": 15}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_delay_delta(self):
        """Test calcul delta délai."""
        manager = ShadowModeManager()

        human_decision = {"delay_minutes": 5}
        rl_decision = {"delay_minutes": 2}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["delay_delta"] == -3  # 2 - 5 = -3

    def test_calculate_kpis_with_missing_values(self):
        """Test calcul KPIs avec valeurs manquantes."""
        manager = ShadowModeManager()

        human_decision = {}
        rl_decision = {}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == 0  # 0 - 0 = 0
        assert kpis["delay_delta"] == 0  # 0 - 0 = 0

    def test_calculate_kpis_with_none_values(self):
        """Test calcul KPIs avec valeurs None."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": None, "delay_minutes": None}
        rl_decision = {"eta_minutes": None, "delay_minutes": None}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == 0  # None - None = 0
        assert kpis["delay_delta"] == 0  # None - None = 0

    def test_calculate_kpis_with_string_values(self):
        """Test calcul KPIs avec valeurs string."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": "15", "delay_minutes": "5"}
        rl_decision = {"eta_minutes": "12", "delay_minutes": "2"}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -3  # 12 - 15 = -3
        assert kpis["delay_delta"] == -3  # 2 - 5 = -3

    def test_calculate_kpis_with_float_values(self):
        """Test calcul KPIs avec valeurs float."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 15.5, "delay_minutes": 2.7}
        rl_decision = {"eta_minutes": 12.3, "delay_minutes": 1.8}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -3.2  # 12.3 - 15.5 = -3.2
        assert kpis["delay_delta"] == -0.9  # 1.8 - 2.7 = -0.9

    def test_calculate_kpis_with_negative_values(self):
        """Test calcul KPIs avec valeurs négatives."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": -5, "delay_minutes": -2}
        rl_decision = {"eta_minutes": -3, "delay_minutes": -1}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == 2  # -3 - (-5) = 2
        assert kpis["delay_delta"] == 1  # -1 - (-2) = 1

    def test_calculate_kpis_with_large_values(self):
        """Test calcul KPIs avec valeurs importantes."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 1000, "delay_minutes": 100}
        rl_decision = {"eta_minutes": 500, "delay_minutes": 50}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -500  # 500 - 1000 = -500
        assert kpis["delay_delta"] == -50  # 50 - 100 = -50

    def test_calculate_kpis_with_zero_values(self):
        """Test calcul KPIs avec valeurs zéro."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 0, "delay_minutes": 0}
        rl_decision = {"eta_minutes": 0, "delay_minutes": 0}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == 0  # 0 - 0 = 0
        assert kpis["delay_delta"] == 0  # 0 - 0 = 0

    def test_calculate_kpis_with_mixed_types(self):
        """Test calcul KPIs avec types mixtes."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 15, "delay_minutes": "5"}
        rl_decision = {"eta_minutes": "12", "delay_minutes": 2}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -3  # 12 - 15 = -3
        assert kpis["delay_delta"] == -3  # 2 - 5 = -3

    def test_calculate_kpis_with_invalid_types(self):
        """Test calcul KPIs avec types invalides."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": [15], "delay_minutes": {"value": 5}}
        rl_decision = {"eta_minutes": [12], "delay_minutes": {"value": 2}}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        # Devrait gérer les types invalides gracieusement
        assert isinstance(kpis["eta_delta"], (int, float))
        assert isinstance(kpis["delay_delta"], (int, float))

    def test_calculate_kpis_with_context(self):
        """Test calcul KPIs avec contexte."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 15}
        rl_decision = {"eta_minutes": 12}
        context = {
            "booking_id": 1,
            "pickup_time": datetime.now(),
            "distance_km": 5.0,
            "traffic_level": "high",
            "weather": "rain"
        }

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis

    def test_calculate_kpis_with_empty_context(self):
        """Test calcul KPIs avec contexte vide."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 15}
        rl_decision = {"eta_minutes": 12}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis

    def test_calculate_kpis_with_none_context(self):
        """Test calcul KPIs avec contexte None."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 15}
        rl_decision = {"eta_minutes": 12}
        context = None

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis

    def test_calculate_kpis_with_complex_decisions(self):
        """Test calcul KPIs avec décisions complexes."""
        manager = ShadowModeManager()

        human_decision = {
            "driver_id": 1,
            "eta_minutes": 20,
            "delay_minutes": 5,
            "confidence": 0.8,
            "reason": "Closest driver"
        }

        rl_decision = {
            "driver_id": 2,
            "eta_minutes": 15,
            "delay_minutes": 2,
            "confidence": 0.9,
            "reason": "Better traffic route"
        }

        context = {
            "booking_id": 1,
            "pickup_time": datetime.now(),
            "distance_km": 5.0,
            "traffic_level": "high",
            "weather": "rain",
            "time_of_day": "rush_hour"
        }

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == -5  # 15 - 20 = -5
        assert kpis["delay_delta"] == -3  # 2 - 5 = -3

    def test_calculate_kpis_with_boolean_values(self):
        """Test calcul KPIs avec valeurs booléennes."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": True, "delay_minutes": False}
        rl_decision = {"eta_minutes": False, "delay_minutes": True}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -1  # False - True = -1
        assert kpis["delay_delta"] == 1  # True - False = 1

    def test_calculate_kpis_with_empty_strings(self):
        """Test calcul KPIs avec chaînes vides."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": "", "delay_minutes": ""}
        rl_decision = {"eta_minutes": "", "delay_minutes": ""}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == 0  # "" - "" = 0
        assert kpis["delay_delta"] == 0  # "" - "" = 0

    def test_calculate_kpis_with_whitespace_strings(self):
        """Test calcul KPIs avec chaînes avec espaces."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": " 15 ", "delay_minutes": " 5 "}
        rl_decision = {"eta_minutes": " 12 ", "delay_minutes": " 2 "}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -3  # 12 - 15 = -3
        assert kpis["delay_delta"] == -3  # 2 - 5 = -3

    def test_calculate_kpis_with_scientific_notation(self):
        """Test calcul KPIs avec notation scientifique."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 1e2, "delay_minutes": 1e1}
        rl_decision = {"eta_minutes": 5e1, "delay_minutes": 5e0}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -50.0  # 50 - 100 = -50
        assert kpis["delay_delta"] == -5.0  # 5 - 10 = -5

    def test_calculate_kpis_with_infinity_values(self):
        """Test calcul KPIs avec valeurs infinies."""
        import math

        manager = ShadowModeManager()

        human_decision = {"eta_minutes": math.inf, "delay_minutes": -math.inf}
        rl_decision = {"eta_minutes": math.inf, "delay_minutes": -math.inf}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == 0  # inf - inf = 0
        assert kpis["delay_delta"] == 0  # -inf - (-inf) = 0

    def test_calculate_kpis_with_nan_values(self):
        """Test calcul KPIs avec valeurs NaN."""

        manager = ShadowModeManager()

        human_decision = {"eta_minutes": math.nan, "delay_minutes": math.nan}
        rl_decision = {"eta_minutes": math.nan, "delay_minutes": math.nan}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        # NaN - NaN devrait être géré gracieusement
        assert isinstance(kpis["eta_delta"], (int, float))
        assert isinstance(kpis["delay_delta"], (int, float))

    def test_calculate_kpis_with_extreme_values(self):
        """Test calcul KPIs avec valeurs extrêmes."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 999999, "delay_minutes": 0.0001}
        rl_decision = {"eta_minutes": 0.0001, "delay_minutes": 999999}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -999998.999  # 0.0001 - 999999 = -999998.999
        assert kpis["delay_delta"] == 999998.999  # 999999 - 0.0001 = 999998.999

    def test_calculate_kpis_with_unicode_values(self):
        """Test calcul KPIs avec valeurs Unicode."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": "15", "delay_minutes": "5"}
        rl_decision = {"eta_minutes": "12", "delay_minutes": "2"}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -3  # 12 - 15 = -3
        assert kpis["delay_delta"] == -3  # 2 - 5 = -3

    def test_calculate_kpis_with_special_characters(self):
        """Test calcul KPIs avec caractères spéciaux."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": "15.5", "delay_minutes": "5.7"}
        rl_decision = {"eta_minutes": "12.3", "delay_minutes": "2.8"}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -3.2  # 12.3 - 15.5 = -3.2
        assert kpis["delay_delta"] == -2.9  # 2.8 - 5.7 = -2.9

    def test_calculate_kpis_with_very_small_values(self):
        """Test calcul KPIs avec valeurs très petites."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 1e-10, "delay_minutes": 1e-15}
        rl_decision = {"eta_minutes": 1e-11, "delay_minutes": 1e-16}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -9e-11  # 1e-11 - 1e-10 = -9e-11
        assert kpis["delay_delta"] == -9e-16  # 1e-16 - 1e-15 = -9e-16

    def test_calculate_kpis_with_very_large_values(self):
        """Test calcul KPIs avec valeurs très importantes."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 1e10, "delay_minutes": 1e15}
        rl_decision = {"eta_minutes": 1e9, "delay_minutes": 1e14}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -9e9  # 1e9 - 1e10 = -9e9
        assert kpis["delay_delta"] == -9e14  # 1e14 - 1e15 = -9e14

    def test_calculate_kpis_with_mixed_precision(self):
        """Test calcul KPIs avec précision mixte."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 15.123456789, "delay_minutes": 5.987654321}
        rl_decision = {"eta_minutes": 12.111111111, "delay_minutes": 2.999999999}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert abs(kpis["eta_delta"] - (-3.012345678)) < 1e-9
        assert abs(kpis["delay_delta"] - (-2.987654322)) < 1e-9

    def test_calculate_kpis_with_complex_scenario(self):
        """Test calcul KPIs avec scénario complexe."""
        manager = ShadowModeManager()

        human_decision = {
            "driver_id": 1,
            "eta_minutes": 20.5,
            "delay_minutes": 3.7,
            "confidence": 0.85,
            "reason": "Driver is closest",
            "traffic_factor": 1.2,
            "weather_factor": 1.1
        }

        rl_decision = {
            "driver_id": 2,
            "eta_minutes": 17.3,
            "delay_minutes": 2.1,
            "confidence": 0.92,
            "reason": "Better route optimization",
            "traffic_factor": 1.0,
            "weather_factor": 1.0
        }

        context = {
            "booking_id": 1,
            "pickup_time": datetime.now(),
            "distance_km": 5.0,
            "traffic_level": "high",
            "weather": "rain",
            "time_of_day": "rush_hour",
            "driver_experience": "expert",
            "vehicle_type": "sedan"
        }

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -3.2  # 17.3 - 20.5 = -3.2
        assert kpis["delay_delta"] == -1.6  # 2.1 - 3.7 = -1.6
