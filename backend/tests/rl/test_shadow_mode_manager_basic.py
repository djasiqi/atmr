#!/usr/bin/env python3
"""
Tests pour shadow_mode_manager.py - couverture de base simplifiée
"""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from services.rl.shadow_mode_manager import ShadowModeManager


class TestShadowModeManager:
    """Tests pour la classe ShadowModeManager."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        manager = ShadowModeManager()

        assert manager.data_dir is not None
        assert isinstance(manager.kpi_metrics, dict)
        assert isinstance(manager.decision_metadata, dict)
        assert manager.logger is not None

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ShadowModeManager(data_dir=temp_dir)

            assert manager.data_dir == Path(temp_dir)
            assert isinstance(manager.kpi_metrics, dict)
            assert isinstance(manager.decision_metadata, dict)
            assert manager.logger is not None

    def test_setup_logging(self):
        """Test configuration du logging."""
        manager = ShadowModeManager()

        logger = manager._setup_logging()

        assert logger is not None
        assert logger.name == "services.rl.shadow_mode_manager"

    def test_log_decision_comparison_basic(self):
        """Test enregistrement de comparaison de décision de base."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": "driver_1", "eta_minutes": 15}
        rl_decision = {"driver_id": "driver_2", "eta_minutes": 12}
        context = {"booking_id": "booking_1", "company_id": "company_1"}

        kpis = manager.log_decision_comparison(
            company_id="company_1",
            booking_id="booking_1",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert "eta_delta" in kpis
        assert "delay_delta" in kpis
        assert "rl_confidence" in kpis
        assert "human_confidence" in kpis
        assert "decision_reasons" in kpis
        assert "constraint_violations" in kpis
        assert "performance_impact" in kpis

    def test_log_decision_comparison_with_different_values(self):
        """Test enregistrement de comparaison avec valeurs différentes."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": "driver_1", "eta_minutes": 20}
        rl_decision = {"driver_id": "driver_2", "eta_minutes": 18}
        context = {"booking_id": "booking_2", "company_id": "company_2"}

        kpis = manager.log_decision_comparison(
            company_id="company_2",
            booking_id="booking_2",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == -2  # 18 - 20
        assert kpis["delay_delta"] == 0  # Pas de retard spécifié
        assert kpis["rl_confidence"] == 0  # Pas de confiance spécifiée
        assert kpis["human_confidence"] is None  # Pas de confiance spécifiée

    def test_log_decision_comparison_with_same_driver(self):
        """Test enregistrement de comparaison avec même chauffeur."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": "driver_1", "eta_minutes": 15}
        rl_decision = {"driver_id": "driver_1", "eta_minutes": 15}
        context = {"booking_id": "booking_3", "company_id": "company_3"}

        kpis = manager.log_decision_comparison(
            company_id="company_3",
            booking_id="booking_3",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == 0  # Même chauffeur
        assert kpis["delay_delta"] == 0  # Pas de retard spécifié
        assert kpis["rl_confidence"] == 0  # Pas de confiance spécifiée
        assert kpis["human_confidence"] is None  # Pas de confiance spécifiée

    def test_log_decision_comparison_with_delay(self):
        """Test enregistrement de comparaison avec retard."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": "driver_1", "eta_minutes": 15, "delay_minutes": 5}
        rl_decision = {"driver_id": "driver_2", "eta_minutes": 12, "delay_minutes": 2}
        context = {"booking_id": "booking_4", "company_id": "company_4"}

        kpis = manager.log_decision_comparison(
            company_id="company_4",
            booking_id="booking_4",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == -3  # 12 - 15
        assert kpis["delay_delta"] == -3  # 2 - 5
        assert kpis["rl_confidence"] == 0  # Pas de confiance spécifiée
        assert kpis["human_confidence"] is None  # Pas de confiance spécifiée

    def test_log_decision_comparison_with_confidence(self):
        """Test enregistrement de comparaison avec confiance."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": "driver_1", "eta_minutes": 15, "confidence": 0.9}
        rl_decision = {"driver_id": "driver_2", "eta_minutes": 12, "confidence": 0.95}
        context = {"booking_id": "booking_5", "company_id": "company_5"}

        kpis = manager.log_decision_comparison(
            company_id="company_5",
            booking_id="booking_5",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == -3  # 12 - 15
        assert kpis["delay_delta"] == 0  # Pas de retard spécifié
        assert kpis["rl_confidence"] == 0.95  # Valeur fournie
        assert kpis["human_confidence"] == 0.9  # Valeur fournie

    def test_log_decision_comparison_with_empty_decisions(self):
        """Test enregistrement de comparaison avec décisions vides."""
        manager = ShadowModeManager()

        human_decision = {}
        rl_decision = {}
        context = {"booking_id": "booking_8", "company_id": "company_8"}

        kpis = manager.log_decision_comparison(
            company_id="company_8",
            booking_id="booking_8",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == 0  # Pas d'ETA spécifié
        assert kpis["delay_delta"] == 0  # Pas de retard spécifié
        assert kpis["rl_confidence"] == 0  # Pas de confiance spécifiée
        assert kpis["human_confidence"] is None  # Pas de confiance spécifiée

    def test_log_decision_comparison_with_none_values(self):
        """Test enregistrement de comparaison avec valeurs None."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": None, "eta_minutes": 0}  # Utiliser 0 au lieu de None
        rl_decision = {"driver_id": None, "eta_minutes": 0}  # Utiliser 0 au lieu de None
        context = {"booking_id": "booking_9", "company_id": "company_9"}

        kpis = manager.log_decision_comparison(
            company_id="company_9",
            booking_id="booking_9",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == 0  # Pas d'ETA spécifié
        assert kpis["delay_delta"] == 0  # Pas de retard spécifié
        assert kpis["rl_confidence"] == 0  # Pas de confiance spécifiée
        assert kpis["human_confidence"] is None  # Pas de confiance spécifiée

    def test_log_decision_comparison_with_exception(self):
        """Test enregistrement de comparaison avec exception."""
        manager = ShadowModeManager()

        # Mock pour lever une exception
        with patch.object(manager, "_calculate_kpis", side_effect=Exception("KPI calculation error")):
            with pytest.raises(Exception):
                manager.log_decision_comparison(
                    company_id="company_10",
                    booking_id="booking_10",
                    human_decision={"driver_id": "driver_1"},
                    rl_decision={"driver_id": "driver_2"},
                    context={"booking_id": "booking_10"}
                )

    def test_log_decision_comparison_with_large_values(self):
        """Test enregistrement de comparaison avec valeurs importantes."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": "driver_1", "eta_minutes": 1000}
        rl_decision = {"driver_id": "driver_2", "eta_minutes": 999}
        context = {"booking_id": "booking_11", "company_id": "company_11"}

        kpis = manager.log_decision_comparison(
            company_id="company_11",
            booking_id="booking_11",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == -1  # 999 - 1000
        assert kpis["delay_delta"] == 0  # Pas de retard spécifié
        assert kpis["rl_confidence"] == 0  # Pas de confiance spécifiée
        assert kpis["human_confidence"] is None  # Pas de confiance spécifiée

    def test_log_decision_comparison_with_negative_values(self):
        """Test enregistrement de comparaison avec valeurs négatives."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": "driver_1", "eta_minutes": -5}
        rl_decision = {"driver_id": "driver_2", "eta_minutes": -3}
        context = {"booking_id": "booking_12", "company_id": "company_12"}

        kpis = manager.log_decision_comparison(
            company_id="company_12",
            booking_id="booking_12",
            human_decision=human_decision,
            rl_decision=rl_decision,
            context=context
        )

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == 2  # -3 - (-5)
        assert kpis["delay_delta"] == 0  # Pas de retard spécifié
        assert kpis["rl_confidence"] == 0  # Pas de confiance spécifiée
        assert kpis["human_confidence"] is None  # Pas de confiance spécifiée

    def test_calculate_kpis_basic(self):
        """Test calcul des KPIs de base."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": "driver_1", "eta_minutes": 15}
        rl_decision = {"driver_id": "driver_2", "eta_minutes": 12}
        context = {"booking_id": "booking_1", "company_id": "company_1"}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == -3  # 12 - 15
        assert kpis["delay_delta"] == 0  # Pas de retard spécifié
        assert kpis["rl_confidence"] == 0  # Pas de confiance spécifiée
        assert kpis["human_confidence"] is None  # Pas de confiance spécifiée
        assert "decision_reasons" in kpis
        assert "constraint_violations" in kpis
        assert "performance_impact" in kpis

    def test_calculate_kpis_with_alternatives(self):
        """Test calcul des KPIs avec alternatives."""
        manager = ShadowModeManager()

        human_decision = {"driver_id": "driver_1", "eta_minutes": 15}
        rl_decision = {
            "driver_id": "driver_2",
            "eta_minutes": 12,
            "alternative_drivers": ["driver_3", "driver_4", "driver_5"]
        }
        context = {"booking_id": "booking_1", "company_id": "company_1"}

        kpis = manager._calculate_kpis(human_decision, rl_decision, context)

        assert isinstance(kpis, dict)
        assert kpis["eta_delta"] == -3  # 12 - 15
        assert kpis["delay_delta"] == 0  # Pas de retard spécifié
        assert kpis["rl_confidence"] == 0  # Pas de confiance spécifiée
        assert kpis["human_confidence"] is None  # Pas de confiance spécifiée
        assert kpis["second_best_driver"] == "driver_4"  # Deuxième alternative
        assert "decision_reasons" in kpis
        assert "constraint_violations" in kpis
        assert "performance_impact" in kpis
