"""Tests simples et corrects pour shadow_mode_manager.py."""

from datetime import datetime

from services.rl.shadow_mode_manager import ShadowModeManager


class TestShadowModeManagerSimpleCorrect:
    """Tests simples et corrects pour ShadowModeManager."""

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

        human_decision = {"driver_id": 1, "eta_minutes": 15, "delay_minutes": 0}

        rl_decision = {"driver_id": 2, "eta_minutes": 12, "delay_minutes": 0}

        context = {"booking_id": 1, "pickup_time": datetime.now(), "distance_km": 5.0}

        kpis = manager.log_decision_comparison(
            company_id="1", booking_id="1", human_decision=human_decision, rl_decision=rl_decision, context=context
        )

        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis
        assert len(manager.decision_metadata["timestamp"]) == 1
        assert manager.decision_metadata["company_id"][0] == "1"
        assert manager.decision_metadata["booking_id"][0] == "1"

    def test_calculate_kpis_basic(self):
        """Test calcul KPIs basique."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": 1, "eta_minutes": 15, "delay_minutes": 0}

        rl_decision = {"driver_id": 2, "eta_minutes": 12, "delay_minutes": 0}

        context = {"booking_id": 1, "pickup_time": datetime.now(), "distance_km": 5.0}

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

    def test_calculate_kpis_with_zero_values(self):
        """Test calcul KPIs avec valeurs zéro."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 0, "delay_minutes": 0}
        rl_decision = {"eta_minutes": 0, "delay_minutes": 0}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == 0  # 0 - 0 = 0
        assert kpis["delay_delta"] == 0  # 0 - 0 = 0

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
            "weather": "rain",
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

    def test_calculate_kpis_with_complex_decisions(self):
        """Test calcul KPIs avec décisions complexes."""
        manager = ShadowModeManager()

        human_decision = {
            "driver_id": 1,
            "eta_minutes": 20,
            "delay_minutes": 5,
            "confidence": 0.8,
            "reason": "Closest driver",
        }

        rl_decision = {
            "driver_id": 2,
            "eta_minutes": 15,
            "delay_minutes": 2,
            "confidence": 0.9,
            "reason": "Better traffic route",
        }

        context = {
            "booking_id": 1,
            "pickup_time": datetime.now(),
            "distance_km": 5.0,
            "traffic_level": "high",
            "weather": "rain",
            "time_of_day": "rush_hour",
        }

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == -5  # 15 - 20 = -5
        assert kpis["delay_delta"] == -3  # 2 - 5 = -3

    def test_calculate_kpis_with_float_values(self):
        """Test calcul KPIs avec valeurs float."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 15.5, "delay_minutes": 2.7}
        rl_decision = {"eta_minutes": 12.3, "delay_minutes": 1.8}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert abs(kpis["eta_delta"] - (-3.2)) < 0.01  # 12.3 - 15.5 = -3.2
        assert abs(kpis["delay_delta"] - (-0.9)) < 0.01  # 1.8 - 2.7 = -0.9

    def test_calculate_kpis_with_same_values(self):
        """Test calcul KPIs avec valeurs identiques."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 15, "delay_minutes": 5}
        rl_decision = {"eta_minutes": 15, "delay_minutes": 5}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == 0  # 15 - 15 = 0
        assert kpis["delay_delta"] == 0  # 5 - 5 = 0

    def test_calculate_kpis_with_different_drivers(self):
        """Test calcul KPIs avec conducteurs différents."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": 1, "eta_minutes": 20}
        rl_decision = {"driver_id": 2, "eta_minutes": 15}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_with_same_driver(self):
        """Test calcul KPIs avec même conducteur."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": 1, "eta_minutes": 20}
        rl_decision = {"driver_id": 1, "eta_minutes": 15}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_with_confidence_values(self):
        """Test calcul KPIs avec valeurs de confiance."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20, "confidence": 0.8}
        rl_decision = {"eta_minutes": 15, "confidence": 0.9}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5
        assert "rl_confidence" in kpis
        assert "human_confidence" in kpis

    def test_calculate_kpis_with_reason_values(self):
        """Test calcul KPIs avec valeurs de raison."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20, "reason": "Closest driver"}
        rl_decision = {"eta_minutes": 15, "reason": "Better route"}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5
        assert "decision_reasons" in kpis

    def test_calculate_kpis_with_constraint_values(self):
        """Test calcul KPIs avec valeurs de contrainte."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20, "constraint_violations": 0}
        rl_decision = {"eta_minutes": 15, "constraint_violations": 1}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5
        assert "constraint_violations" in kpis

    def test_calculate_kpis_with_performance_values(self):
        """Test calcul KPIs avec valeurs de performance."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20, "performance_impact": 0.1}
        rl_decision = {"eta_minutes": 15, "performance_impact": 0.2}
        context = {}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5
        assert "performance_impact" in kpis

    def test_calculate_kpis_with_second_best_driver(self):
        """Test calcul KPIs avec second meilleur conducteur."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20, "driver_id": 1}
        rl_decision = {"eta_minutes": 15, "driver_id": 2}
        context = {"second_best_driver": 3}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5
        assert "second_best_driver" in kpis

    def test_calculate_kpis_with_traffic_context(self):
        """Test calcul KPIs avec contexte de trafic."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20}
        rl_decision = {"eta_minutes": 15}
        context = {"traffic_level": "high", "avg_eta": 18, "traffic_factor": 1.2}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_with_weather_context(self):
        """Test calcul KPIs avec contexte météo."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20}
        rl_decision = {"eta_minutes": 15}
        context = {"weather": "rain", "weather_factor": 1.1, "visibility": "poor"}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_with_time_context(self):
        """Test calcul KPIs avec contexte temporel."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20}
        rl_decision = {"eta_minutes": 15}
        context = {"time_of_day": "rush_hour", "day_of_week": "monday", "hour": 17}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_with_distance_context(self):
        """Test calcul KPIs avec contexte de distance."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20}
        rl_decision = {"eta_minutes": 15}
        context = {
            "distance_km": 5.0,
            "pickup_lat": 48.8566,
            "pickup_lon": 2.3522,
            "dropoff_lat": 48.8606,
            "dropoff_lon": 2.3376,
        }

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_with_vehicle_context(self):
        """Test calcul KPIs avec contexte véhicule."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20}
        rl_decision = {"eta_minutes": 15}
        context = {"vehicle_type": "sedan", "vehicle_capacity": 4, "vehicle_fuel_type": "gasoline"}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_with_driver_context(self):
        """Test calcul KPIs avec contexte conducteur."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20}
        rl_decision = {"eta_minutes": 15}
        context = {"driver_experience": "expert", "driver_rating": 4.8, "driver_availability": "high"}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_with_booking_context(self):
        """Test calcul KPIs avec contexte réservation."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20}
        rl_decision = {"eta_minutes": 15}
        context = {"booking_id": 1, "pickup_time": datetime.now(), "booking_type": "standard", "priority": "normal"}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_with_company_context(self):
        """Test calcul KPIs avec contexte entreprise."""
        manager = ShadowModeManager()

        human_decision = {"eta_minutes": 20}
        rl_decision = {"eta_minutes": 15}
        context = {"company_id": 1, "company_size": "large", "company_type": "taxi"}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5

    def test_calculate_kpis_with_all_context(self):
        """Test calcul KPIs avec tout le contexte."""
        manager = ShadowModeManager()

        human_decision = {
            "driver_id": 1,
            "eta_minutes": 20,
            "delay_minutes": 5,
            "confidence": 0.8,
            "reason": "Closest driver",
        }

        rl_decision = {
            "driver_id": 2,
            "eta_minutes": 15,
            "delay_minutes": 2,
            "confidence": 0.9,
            "reason": "Better route",
        }

        context = {
            "booking_id": 1,
            "pickup_time": datetime.now(),
            "distance_km": 5.0,
            "traffic_level": "high",
            "weather": "rain",
            "time_of_day": "rush_hour",
            "vehicle_type": "sedan",
            "driver_experience": "expert",
            "company_id": 1,
            "avg_eta": 18,
            "second_best_driver": 3,
        }

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert kpis["eta_delta"] == -5  # 15 - 20 = -5
        assert kpis["delay_delta"] == -3  # 2 - 5 = -3
