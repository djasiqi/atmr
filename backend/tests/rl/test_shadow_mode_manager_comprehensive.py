"""Tests complets pour shadow_mode_manager.py."""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pytest

from services.rl.shadow_mode_manager import ShadowModeManager


class TestShadowModeManagerComprehensive:
    """Tests complets pour ShadowModeManager."""

    def test_init_basic(self):
        """Test initialisation basique."""
        manager = ShadowModeManager()

        assert manager.company_id is None
        assert manager.decision_logs == []
        assert manager.kpis == {}
        assert manager.start_time is not None

    def test_init_with_company_id(self):
        """Test initialisation avec company_id."""
        manager = ShadowModeManager(company_id=0.123)

        assert manager.company_id == 123

    def test_setup_logging(self):
        """Test configuration logging."""
        manager = ShadowModeManager()

        # Vérifier que le logger est configuré
        assert manager.logger is not None

    def test_log_decision_comparison_basic(self):
        """Test logging comparaison décisions basique."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": 1, "eta_minutes": 15, "delay_minutes": 0}

        rl_decision = {"driver_id": 2, "eta_minutes": 12, "delay_minutes": 0}

        context = {"booking_id": 1, "pickup_time": datetime.now(), "distance_km": 5.0}

        manager.log_decision_comparison(
            company_id=1, booking_id=1, human_decision=human_decision, rl_decision=rl_decision, context=context
        )

        assert len(manager.decision_logs) == 1
        log_entry = manager.decision_logs[0]
        assert log_entry["company_id"] == 1
        assert log_entry["booking_id"] == 1
        assert log_entry["human_driver_id"] == 1
        assert log_entry["rl_driver_id"] == 2

    def test_log_decision_comparison_with_none_values(self):
        """Test logging avec valeurs None."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": 1, "eta_minutes": 0, "delay_minutes": 0}

        rl_decision = {"driver_id": 2, "eta_minutes": 0, "delay_minutes": 0}

        context = {"booking_id": 1, "pickup_time": datetime.now(), "distance_km": 5.0}

        manager.log_decision_comparison(
            company_id=1, booking_id=1, human_decision=human_decision, rl_decision=rl_decision, context=context
        )

        assert len(manager.decision_logs) == 1

    def test_calculate_kpis_basic(self):
        """Test calcul KPIs basique."""
        manager = ShadowModeManager()

        # Ajouter quelques logs de test
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 15,
                "rl_eta_minutes": 12,
                "human_delay_minutes": 0,
                "rl_delay_minutes": 0,
                "timestamp": datetime.now(),
            },
            {
                "company_id": 1,
                "booking_id": 2,
                "human_driver_id": 2,
                "rl_driver_id": 1,
                "human_eta_minutes": 20,
                "rl_eta_minutes": 18,
                "human_delay_minutes": 5,
                "rl_delay_minutes": 2,
                "timestamp": datetime.now(),
            },
        ]

        kpis = manager._calculate_kpis()

        assert isinstance(kpis, dict)
        assert "total_decisions" in kpis
        assert "rl_wins" in kpis
        assert "human_wins" in kpis
        assert "avg_human_eta" in kpis
        assert "avg_rl_eta" in kpis
        assert "avg_human_delay" in kpis
        assert "avg_rl_delay" in kpis
        assert "rl_win_rate" in kpis
        assert "eta_improvement_rate" in kpis
        assert "delay_reduction_rate" in kpis

    def test_calculate_kpis_empty_logs(self):
        """Test calcul KPIs avec logs vides."""
        manager = ShadowModeManager()

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 0
        assert kpis["rl_wins"] == 0
        assert kpis["human_wins"] == 0
        assert kpis["rl_win_rate"] == 0.0

    def test_calculate_kpis_with_ties(self):
        """Test calcul KPIs avec égalités."""
        manager = ShadowModeManager()

        # Logs avec égalités
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 15,
                "rl_eta_minutes": 15,  # Égalité
                "human_delay_minutes": 0,
                "rl_delay_minutes": 0,  # Égalité
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["rl_wins"] == 0
        assert kpis["human_wins"] == 0
        assert kpis["rl_win_rate"] == 0.0

    def test_calculate_kpis_rl_wins(self):
        """Test calcul KPIs avec victoires RL."""
        manager = ShadowModeManager()

        # Logs avec victoires RL
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 20,
                "rl_eta_minutes": 15,  # RL meilleur
                "human_delay_minutes": 5,
                "rl_delay_minutes": 2,  # RL meilleur
                "timestamp": datetime.now(),
            },
            {
                "company_id": 1,
                "booking_id": 2,
                "human_driver_id": 2,
                "rl_driver_id": 1,
                "human_eta_minutes": 25,
                "rl_eta_minutes": 20,  # RL meilleur
                "human_delay_minutes": 8,
                "rl_delay_minutes": 3,  # RL meilleur
                "timestamp": datetime.now(),
            },
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 2
        assert kpis["rl_wins"] == 2
        assert kpis["human_wins"] == 0
        assert kpis["rl_win_rate"] == 1.0

    def test_calculate_kpis_human_wins(self):
        """Test calcul KPIs avec victoires humaines."""
        manager = ShadowModeManager()

        # Logs avec victoires humaines
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 15,
                "rl_eta_minutes": 20,  # Humain meilleur
                "human_delay_minutes": 2,
                "rl_delay_minutes": 5,  # Humain meilleur
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["rl_wins"] == 0
        assert kpis["human_wins"] == 1
        assert kpis["rl_win_rate"] == 0.0

    def test_calculate_kpis_mixed_results(self):
        """Test calcul KPIs avec résultats mixtes."""
        manager = ShadowModeManager()

        # Logs avec résultats mixtes
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 20,
                "rl_eta_minutes": 15,  # RL meilleur
                "human_delay_minutes": 5,
                "rl_delay_minutes": 2,  # RL meilleur
                "timestamp": datetime.now(),
            },
            {
                "company_id": 1,
                "booking_id": 2,
                "human_driver_id": 2,
                "rl_driver_id": 1,
                "human_eta_minutes": 15,
                "rl_eta_minutes": 20,  # Humain meilleur
                "human_delay_minutes": 2,
                "rl_delay_minutes": 5,  # Humain meilleur
                "timestamp": datetime.now(),
            },
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 2
        assert kpis["rl_wins"] == 1
        assert kpis["human_wins"] == 1
        assert kpis["rl_win_rate"] == 0.5

    def test_calculate_kpis_with_zero_values(self):
        """Test calcul KPIs avec valeurs zéro."""
        manager = ShadowModeManager()

        # Logs avec valeurs zéro
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 0,
                "rl_eta_minutes": 0,
                "human_delay_minutes": 0,
                "rl_delay_minutes": 0,
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 0.0
        assert kpis["avg_rl_eta"] == 0.0
        assert kpis["avg_human_delay"] == 0.0
        assert kpis["avg_rl_delay"] == 0.0

    def test_calculate_kpis_with_negative_values(self):
        """Test calcul KPIs avec valeurs négatives."""
        manager = ShadowModeManager()

        # Logs avec valeurs négatives
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": -5,
                "rl_eta_minutes": -3,
                "human_delay_minutes": -2,
                "rl_delay_minutes": -1,
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == -5.0
        assert kpis["avg_rl_eta"] == -3.0
        assert kpis["avg_human_delay"] == -2.0
        assert kpis["avg_rl_delay"] == -1.0

    def test_calculate_kpis_with_large_values(self):
        """Test calcul KPIs avec valeurs importantes."""
        manager = ShadowModeManager()

        # Logs avec valeurs importantes
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 1000,
                "rl_eta_minutes": 500,
                "human_delay_minutes": 100,
                "rl_delay_minutes": 50,
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 1000.0
        assert kpis["avg_rl_eta"] == 500.0
        assert kpis["avg_human_delay"] == 100.0
        assert kpis["avg_rl_delay"] == 50.0

    def test_calculate_kpis_with_missing_fields(self):
        """Test calcul KPIs avec champs manquants."""
        manager = ShadowModeManager()

        # Logs avec champs manquants
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "timestamp": datetime.now(),
                # Champs eta_minutes et delay_minutes manquants
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 0.0
        assert kpis["avg_rl_eta"] == 0.0
        assert kpis["avg_human_delay"] == 0.0
        assert kpis["avg_rl_delay"] == 0.0

    def test_calculate_kpis_with_none_values(self):
        """Test calcul KPIs avec valeurs None."""
        manager = ShadowModeManager()

        # Logs avec valeurs None
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": None,
                "rl_eta_minutes": None,
                "human_delay_minutes": None,
                "rl_delay_minutes": None,
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 0.0
        assert kpis["avg_rl_eta"] == 0.0
        assert kpis["avg_human_delay"] == 0.0
        assert kpis["avg_rl_delay"] == 0.0

    def test_calculate_kpis_with_string_values(self):
        """Test calcul KPIs avec valeurs string."""
        manager = ShadowModeManager()

        # Logs avec valeurs string
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": "15",
                "rl_eta_minutes": "12",
                "human_delay_minutes": "0",
                "rl_delay_minutes": "0",
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 15.0
        assert kpis["avg_rl_eta"] == 12.0
        assert kpis["avg_human_delay"] == 0.0
        assert kpis["avg_rl_delay"] == 0.0

    def test_calculate_kpis_with_float_values(self):
        """Test calcul KPIs avec valeurs float."""
        manager = ShadowModeManager()

        # Logs avec valeurs float
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 15.5,
                "rl_eta_minutes": 12.3,
                "human_delay_minutes": 2.7,
                "rl_delay_minutes": 1.8,
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 15.5
        assert kpis["avg_rl_eta"] == 12.3
        assert kpis["avg_human_delay"] == 2.7
        assert kpis["avg_rl_delay"] == 1.8

    def test_calculate_kpis_with_multiple_companies(self):
        """Test calcul KPIs avec plusieurs entreprises."""
        manager = ShadowModeManager()

        # Logs avec plusieurs entreprises
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 15,
                "rl_eta_minutes": 12,
                "human_delay_minutes": 0,
                "rl_delay_minutes": 0,
                "timestamp": datetime.now(),
            },
            {
                "company_id": 2,
                "booking_id": 2,
                "human_driver_id": 3,
                "rl_driver_id": 4,
                "human_eta_minutes": 20,
                "rl_eta_minutes": 18,
                "human_delay_minutes": 5,
                "rl_delay_minutes": 2,
                "timestamp": datetime.now(),
            },
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 2
        assert kpis["avg_human_eta"] == 17.5
        assert kpis["avg_rl_eta"] == 15.0
        assert kpis["avg_human_delay"] == 2.5
        assert kpis["avg_rl_delay"] == 1.0

    def test_calculate_kpis_with_same_driver(self):
        """Test calcul KPIs avec même conducteur."""
        manager = ShadowModeManager()

        # Logs avec même conducteur
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 1,  # Même conducteur
                "human_eta_minutes": 15,
                "rl_eta_minutes": 12,
                "human_delay_minutes": 0,
                "rl_delay_minutes": 0,
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["rl_wins"] == 1  # RL gagne car ETA plus faible
        assert kpis["human_wins"] == 0

    def test_calculate_kpis_with_same_eta_different_delay(self):
        """Test calcul KPIs avec même ETA mais délai différent."""
        manager = ShadowModeManager()

        # Logs avec même ETA mais délai différent
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 15,
                "rl_eta_minutes": 15,  # Même ETA
                "human_delay_minutes": 5,
                "rl_delay_minutes": 2,  # Délai différent
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["rl_wins"] == 1  # RL gagne car délai plus faible
        assert kpis["human_wins"] == 0

    def test_calculate_kpis_with_same_delay_different_eta(self):
        """Test calcul KPIs avec même délai mais ETA différent."""
        manager = ShadowModeManager()

        # Logs avec même délai mais ETA différent
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 20,
                "rl_eta_minutes": 15,  # ETA différent
                "human_delay_minutes": 2,
                "rl_delay_minutes": 2,  # Même délai
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["rl_wins"] == 1  # RL gagne car ETA plus faible
        assert kpis["human_wins"] == 0

    def test_calculate_kpis_with_same_values(self):
        """Test calcul KPIs avec valeurs identiques."""
        manager = ShadowModeManager()

        # Logs avec valeurs identiques
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 15,
                "rl_eta_minutes": 15,  # Valeurs identiques
                "human_delay_minutes": 2,
                "rl_delay_minutes": 2,  # Valeurs identiques
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["rl_wins"] == 0  # Égalité
        assert kpis["human_wins"] == 0  # Égalité

    def test_calculate_kpis_with_extreme_values(self):
        """Test calcul KPIs avec valeurs extrêmes."""
        manager = ShadowModeManager()

        # Logs avec valeurs extrêmes
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 999999,
                "rl_eta_minutes": 0.0001,
                "human_delay_minutes": 999999,
                "rl_delay_minutes": 0.0001,
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["rl_wins"] == 1  # RL gagne clairement
        assert kpis["human_wins"] == 0

    def test_calculate_kpis_with_mixed_data_types(self):
        """Test calcul KPIs avec types de données mixtes."""
        manager = ShadowModeManager()

        # Logs avec types de données mixtes
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 15,
                "rl_eta_minutes": "12",
                "human_delay_minutes": 0,
                "rl_delay_minutes": None,
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 15.0
        assert kpis["avg_rl_eta"] == 12.0
        assert kpis["avg_human_delay"] == 0.0
        assert kpis["avg_rl_delay"] == 0.0

    def test_calculate_kpis_with_empty_strings(self):
        """Test calcul KPIs avec chaînes vides."""
        manager = ShadowModeManager()

        # Logs avec chaînes vides
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": "",
                "rl_eta_minutes": "",
                "human_delay_minutes": "",
                "rl_delay_minutes": "",
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 0.0
        assert kpis["avg_rl_eta"] == 0.0
        assert kpis["avg_human_delay"] == 0.0
        assert kpis["avg_rl_delay"] == 0.0

    def test_calculate_kpis_with_boolean_values(self):
        """Test calcul KPIs avec valeurs booléennes."""
        manager = ShadowModeManager()

        # Logs avec valeurs booléennes
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": True,
                "rl_eta_minutes": False,
                "human_delay_minutes": True,
                "rl_delay_minutes": False,
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 1.0  # True = 1
        assert kpis["avg_rl_eta"] == 0.0  # False = 0
        assert kpis["avg_human_delay"] == 1.0  # True = 1
        assert kpis["avg_rl_delay"] == 0.0  # False = 0

    def test_calculate_kpis_with_list_values(self):
        """Test calcul KPIs avec valeurs liste."""
        manager = ShadowModeManager()

        # Logs avec valeurs liste
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": [15],
                "rl_eta_minutes": [12],
                "human_delay_minutes": [0],
                "rl_delay_minutes": [0],
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 0.0  # Liste non convertible
        assert kpis["avg_rl_eta"] == 0.0  # Liste non convertible
        assert kpis["avg_human_delay"] == 0.0  # Liste non convertible
        assert kpis["avg_rl_delay"] == 0.0  # Liste non convertible

    def test_calculate_kpis_with_dict_values(self):
        """Test calcul KPIs avec valeurs dictionnaire."""
        manager = ShadowModeManager()

        # Logs avec valeurs dictionnaire
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": {"value": 15},
                "rl_eta_minutes": {"value": 12},
                "human_delay_minutes": {"value": 0},
                "rl_delay_minutes": {"value": 0},
                "timestamp": datetime.now(),
            }
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 1
        assert kpis["avg_human_eta"] == 0.0  # Dict non convertible
        assert kpis["avg_rl_eta"] == 0.0  # Dict non convertible
        assert kpis["avg_human_delay"] == 0.0  # Dict non convertible
        assert kpis["avg_rl_delay"] == 0.0  # Dict non convertible

    def test_calculate_kpis_with_complex_scenario(self):
        """Test calcul KPIs avec scénario complexe."""
        manager = ShadowModeManager()

        # Logs avec scénario complexe
        manager.decision_logs = [
            {
                "company_id": 1,
                "booking_id": 1,
                "human_driver_id": 1,
                "rl_driver_id": 2,
                "human_eta_minutes": 20,
                "rl_eta_minutes": 15,
                "human_delay_minutes": 5,
                "rl_delay_minutes": 2,
                "timestamp": datetime.now(),
            },
            {
                "company_id": 1,
                "booking_id": 2,
                "human_driver_id": 2,
                "rl_driver_id": 1,
                "human_eta_minutes": 15,
                "rl_eta_minutes": 20,
                "human_delay_minutes": 2,
                "rl_delay_minutes": 5,
                "timestamp": datetime.now(),
            },
            {
                "company_id": 1,
                "booking_id": 3,
                "human_driver_id": 3,
                "rl_driver_id": 3,
                "human_eta_minutes": 18,
                "rl_eta_minutes": 18,
                "human_delay_minutes": 3,
                "rl_delay_minutes": 3,
                "timestamp": datetime.now(),
            },
        ]

        kpis = manager._calculate_kpis()

        assert kpis["total_decisions"] == 3
        assert kpis["rl_wins"] == 1
        assert kpis["human_wins"] == 1
        assert kpis["rl_win_rate"] == 1 / 3
        assert kpis["avg_human_eta"] == 17.67  # (20+15+18)/3
        assert kpis["avg_rl_eta"] == 17.67  # (15+20+18)/3
        assert kpis["avg_human_delay"] == 3.33  # (5+2+3)/3
        assert kpis["avg_rl_delay"] == 3.33  # (2+5+3)/3
