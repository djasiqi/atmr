# ruff: noqa: DTZ001, DTZ003, N803
"""
Tests pour le Shadow Mode Manager.

Vérifie que le shadow mode fonctionne correctement en production.
"""
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.rl.shadow_mode_manager import ShadowModeManager


class TestShadowModeManagerCreation:
    """Tests de création du Shadow Mode Manager."""

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_shadow_manager_creation(self, MockDQNAgent):
        """Test création basique du shadow manager."""
        # Mock l'agent
        mock_agent = MagicMock()
        MockDQNAgent.return_value = mock_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy_model.pth",
                log_dir=tmpdir,
                enable_logging=False
            )

            assert manager is not None
            assert manager.predictions_count == 0
            assert manager.comparisons_count == 0
            assert manager.agreements_count == 0

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_shadow_manager_creates_log_dir(self, MockDQNAgent):
        """Test que le manager crée le répertoire de logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "shadow_logs")

            ShadowModeManager(
                model_path="dummy_model.pth",
                log_dir=log_dir,
                enable_logging=True
            )

            assert os.path.exists(log_dir)
            assert os.path.isdir(log_dir)


class TestShadowModePredictions:
    """Tests des prédictions shadow."""

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_predict_driver_assignment(self, MockDQNAgent):
        """Test prédiction d'assignation."""
        # Mock agent qui retourne une action
        mock_agent = MagicMock()
        mock_agent.select_action.return_value = 0  # Premier driver
        mock_agent.get_q_values.return_value = np.array([100.0, 50.0, 30.0])
        MockDQNAgent.return_value = mock_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy.pth",
                log_dir=tmpdir,
                enable_logging=False
            )
            manager.agent = mock_agent

            # Mock booking et drivers
            mock_booking = MagicMock()
            mock_booking.id = 123

            mock_driver1 = MagicMock()
            mock_driver1.id = 1

            mock_driver2 = MagicMock()
            mock_driver2.id = 2

            # Prédire
            prediction = manager.predict_driver_assignment(
                booking=mock_booking,
                available_drivers=[mock_driver1, mock_driver2],
                current_assignments={}
            )

            assert prediction is not None
            assert prediction['booking_id'] == 123
            assert prediction['predicted_driver_id'] == 1
            assert prediction['action_type'] == 'assign'
            assert 'confidence' in prediction
            assert manager.predictions_count == 1

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_predict_wait_action(self, MockDQNAgent):
        """Test prédiction d'action 'wait'."""
        mock_agent = MagicMock()
        mock_agent.select_action.return_value = 2  # Index > len(drivers)
        mock_agent.get_q_values.return_value = np.array([50.0, 30.0, 100.0])  # Wait est le meilleur
        MockDQNAgent.return_value = mock_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy.pth",
                log_dir=tmpdir,
                enable_logging=False
            )
            manager.agent = mock_agent

            mock_booking = MagicMock()
            mock_booking.id = 456

            mock_driver = MagicMock()
            mock_driver.id = 1

            prediction = manager.predict_driver_assignment(
                booking=mock_booking,
                available_drivers=[mock_driver],  # 1 seul driver
                current_assignments={}
            )

            assert prediction is not None
            assert prediction['booking_id'] == 456
            assert prediction['predicted_driver_id'] is None
            assert prediction['action_type'] == 'wait'


class TestShadowModeComparisons:
    """Tests des comparaisons avec décisions réelles."""

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_compare_agreement(self, MockDQNAgent):
        """Test comparaison avec accord."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy.pth",
                log_dir=tmpdir,
                enable_logging=False
            )

            prediction = {
                "booking_id": 123,
                "predicted_driver_id": 1,
                "confidence": 0.85
            }

            comparison = manager.compare_with_actual_decision(
                prediction=prediction,
                actual_driver_id=1,  # Même driver
                outcome_metrics={"distance_km": 5.2}
            )

            assert comparison['agreement'] is True
            assert comparison['booking_id'] == 123
            assert comparison['predicted_driver_id'] == 1
            assert comparison['actual_driver_id'] == 1
            assert manager.comparisons_count == 1
            assert manager.agreements_count == 1

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_compare_disagreement(self, MockDQNAgent):
        """Test comparaison avec désaccord."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy.pth",
                log_dir=tmpdir,
                enable_logging=False
            )

            prediction = {
                "booking_id": 123,
                "predicted_driver_id": 1,
                "confidence": 0.75
            }

            comparison = manager.compare_with_actual_decision(
                prediction=prediction,
                actual_driver_id=2,  # Driver différent
                outcome_metrics={"distance_km": 3.8}
            )

            assert comparison['agreement'] is False
            assert manager.comparisons_count == 1
            assert manager.agreements_count == 0


class TestShadowModeStats:
    """Tests des statistiques."""

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_get_stats(self, MockDQNAgent):
        """Test récupération des stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy.pth",
                log_dir=tmpdir,
                enable_logging=False
            )

            # Simulations
            manager.predictions_count = 100
            manager.comparisons_count = 95
            manager.agreements_count = 80

            stats = manager.get_stats()

            assert stats['predictions_count'] == 100
            assert stats['comparisons_count'] == 95
            assert stats['agreements_count'] == 80
            assert abs(stats['agreement_rate'] - 80/95) < 0.01

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_agreement_rate_zero_comparisons(self, MockDQNAgent):
        """Test agreement rate quand aucune comparaison."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy.pth",
                log_dir=tmpdir,
                enable_logging=False
            )

            stats = manager.get_stats()
            assert stats['agreement_rate'] == 0.0


class TestShadowModeLogging:
    """Tests du logging."""

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_logging_predictions(self, MockDQNAgent):
        """Test que les prédictions sont loggées."""
        mock_agent = MagicMock()
        mock_agent.select_action.return_value = 0
        mock_agent.get_q_values.return_value = np.array([100.0, 50.0])
        MockDQNAgent.return_value = mock_agent

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy.pth",
                log_dir=tmpdir,
                enable_logging=True  # Logging activé
            )
            manager.agent = mock_agent

            mock_booking = MagicMock()
            mock_booking.id = 789

            mock_driver = MagicMock()
            mock_driver.id = 5

            manager.predict_driver_assignment(
                booking=mock_booking,
                available_drivers=[mock_driver],
                current_assignments={}
            )

            # Vérifier que le fichier de log existe
            from datetime import datetime
            log_file = f"predictions_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            log_path = os.path.join(tmpdir, log_file)

            assert os.path.exists(log_path)

            # Vérifier le contenu
            with open(log_path, encoding='utf-8') as f:
                line = f.readline()
                prediction = json.loads(line)
                assert prediction['booking_id'] == 789
                assert prediction['predicted_driver_id'] == 5

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_logging_comparisons(self, MockDQNAgent):
        """Test que les comparaisons sont loggées."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy.pth",
                log_dir=tmpdir,
                enable_logging=True
            )

            prediction = {
                "booking_id": 999,
                "predicted_driver_id": 3,
                "confidence": 0.92
            }

            manager.compare_with_actual_decision(
                prediction=prediction,
                actual_driver_id=3,
                outcome_metrics={"distance_km": 2.5}
            )

            # Vérifier le fichier
            from datetime import datetime
            log_file = f"comparisons_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
            log_path = os.path.join(tmpdir, log_file)

            assert os.path.exists(log_path)

            with open(log_path, encoding='utf-8') as f:
                line = f.readline()
                comparison = json.loads(line)
                assert comparison['booking_id'] == 999
                assert comparison['agreement'] is True


class TestShadowModeDailyReport:
    """Tests des rapports quotidiens."""

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_generate_daily_report_empty(self, MockDQNAgent):
        """Test génération rapport sans données."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy.pth",
                log_dir=tmpdir,
                enable_logging=False
            )

            report = manager.generate_daily_report()

            assert report is not None
            assert 'summary' in report
            assert report['summary']['total_predictions'] == 0
            assert report['summary']['total_comparisons'] == 0
            assert report['summary']['agreement_rate'] == 0.0

    @patch('services.rl.shadow_mode_manager.DQNAgent')
    def test_daily_report_saves_to_file(self, MockDQNAgent):
        """Test que le rapport est sauvegardé."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ShadowModeManager(
                model_path="dummy.pth",
                log_dir=tmpdir,
                enable_logging=False
            )

            report = manager.generate_daily_report()

            # Vérifier le fichier
            from datetime import datetime
            report_file = f"daily_report_{datetime.utcnow().strftime('%Y%m%d')}.json"
            report_path = os.path.join(tmpdir, report_file)

            assert os.path.exists(report_path)

            # Vérifier le contenu
            with open(report_path, encoding='utf-8') as f:
                saved_report = json.load(f)
                assert saved_report == report

