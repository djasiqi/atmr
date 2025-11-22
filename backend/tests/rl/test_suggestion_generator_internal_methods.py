"""Tests pour les méthodes internes de RLSuggestionGenerator."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from services.rl.suggestion_generator import RLSuggestionGenerator, lazy_import_rl


class TestRLSuggestionGeneratorInternalMethods:
    """Tests pour les méthodes internes de RLSuggestionGenerator."""

    def test_lazy_import_rl_success(self):
        """Test import RL réussi."""
        with (
            patch("services.rl.suggestion_generator._dqn_agent", None),
            patch("services.rl.suggestion_generator._dispatch_env", None),
            patch("services.rl.suggestion_generator.improved_dqn_agent") as mock_dqn,
            patch("services.rl.suggestion_generator.dispatch_env") as mock_env,
        ):
            from services.rl.suggestion_generator import _lazy_import_rl

            _lazy_import_rl()

            assert mock_dqn is not None
            assert mock_env is not None

    def test_lazy_import_rl_failure(self):
        """Test import RL échec."""
        with (
            patch("services.rl.suggestion_generator._dqn_agent", None),
            patch("services.rl.suggestion_generator._dispatch_env", None),
            patch("services.rl.suggestion_generator.improved_dqn_agent", side_effect=ImportError("Test error")),
            pytest.raises(ImportError),
        ):
            lazy_import_rl()

    def test_load_model_success(self):
        """Test chargement modèle réussi."""
        with (
            patch("services.rl.suggestion_generator.Path") as mock_path,
            patch("services.rl.suggestion_generator.DispatchEnv") as mock_env_class,
            patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class,
            patch("services.rl.suggestion_generator._lazy_import_rl"),
        ):
            # Mock file exists
            mock_file = Mock()
            mock_file.exists.return_value = True
            mock_path.return_value = mock_file

            # Mock environment
            mock_env = Mock()
            mock_env.observation_space.shape = [62]
            mock_env.action_space.n = 51
            mock_env_class.return_value = mock_env

            # Mock agent
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            assert generator.agent is not None
            mock_agent.load.assert_called_once()

    def test_load_model_file_not_exists(self):
        """Test chargement modèle - fichier n'existe pas."""
        with (
            patch("services.rl.suggestion_generator.Path") as mock_path,
            patch("services.rl.suggestion_generator._lazy_import_rl"),
        ):
            # Mock file doesn't exist
            mock_file = Mock()
            mock_file.exists.return_value = False
            mock_path.return_value = mock_file

            generator = RLSuggestionGenerator()

            assert generator.agent is None

    def test_load_model_exception(self):
        """Test chargement modèle - exception."""
        with (
            patch("services.rl.suggestion_generator.Path") as mock_path,
            patch("services.rl.suggestion_generator._lazy_import_rl", side_effect=Exception("Test error")),
        ):
            mock_file = Mock()
            mock_file.exists.return_value = True
            mock_path.return_value = mock_file

            generator = RLSuggestionGenerator()

            assert generator.agent is None

    def test_generate_basic_suggestions(self):
        """Test génération suggestions basiques."""
        generator = RLSuggestionGenerator()

        # Mock assignments et drivers
        assignments = [{"id": 1, "driver_id": 1, "booking_id": 1}, {"id": 2, "driver_id": 2, "booking_id": 2}]
        drivers = [
            {"id": 1, "lat": 48.8566, "lon": 2.3522, "available": True},
            {"id": 2, "lat": 48.8606, "lon": 2.3376, "available": True},
        ]

        suggestions = generator._generate_basic_suggestions(assignments=assignments, drivers=drivers, max_suggestions=5)

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5

    def test_generate_basic_suggestions_empty(self):
        """Test génération suggestions basiques - données vides."""
        generator = RLSuggestionGenerator()

        suggestions = generator._generate_basic_suggestions(assignments=[], drivers=[], max_suggestions=5)

        assert suggestions == []

    def test_generate_rl_suggestions(self):
        """Test génération suggestions RL."""
        generator = RLSuggestionGenerator()

        # Mock agent
        mock_agent = Mock()
        mock_q_values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_agent.q_network.return_value = mock_q_values
        generator.agent = mock_agent

        # Mock environment
        mock_env = Mock()
        mock_env.reset.return_value = (np.zeros(62), {})
        mock_env.step.return_value = (np.zeros(62), 0.0, False, False, {})
        generator.env = mock_env

        assignments = [{"id": 1, "driver_id": 1, "booking_id": 1}, {"id": 2, "driver_id": 2, "booking_id": 2}]
        drivers = [
            {"id": 1, "lat": 48.8566, "lon": 2.3522, "available": True},
            {"id": 2, "lat": 48.8606, "lon": 2.3376, "available": True},
        ]

        suggestions = generator._generate_rl_suggestions(
            assignments=assignments, drivers=drivers, min_confidence=0.3, max_suggestions=5
        )

        assert isinstance(suggestions, list)

    def test_generate_rl_suggestions_no_agent(self):
        """Test génération suggestions RL - pas d'agent."""
        generator = RLSuggestionGenerator()
        generator.agent = None

        suggestions = generator._generate_rl_suggestions(
            assignments=[], drivers=[], min_confidence=0.5, max_suggestions=5
        )

        assert suggestions == []

    def test_get_suggestion_confidence(self):
        """Test calcul confiance suggestion."""
        generator = RLSuggestionGenerator()

        # Mock Q-values
        q_values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        confidence = generator._get_suggestion_confidence(q_values, 2)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_format_suggestion(self):
        """Test formatage suggestion."""
        generator = RLSuggestionGenerator()

        suggestion = generator._format_suggestion(
            assignment_id=1, driver_id=2, booking_id=3, confidence=0.8, reason="Test reason"
        )

        assert isinstance(suggestion, dict)
        assert suggestion["assignment_id"] == 1
        assert suggestion["driver_id"] == 2
        assert suggestion["booking_id"] == 3
        assert suggestion["confidence"] == 0.8
        assert suggestion["reason"] == "Test reason"

    def test_get_heuristic_suggestions(self):
        """Test suggestions heuristiques."""
        generator = RLSuggestionGenerator()

        assignments = [
            {"id": 1, "driver_id": 1, "booking_id": 1, "pickup_lat": 48.8566, "pickup_lon": 2.3522},
            {"id": 2, "driver_id": 2, "booking_id": 2, "pickup_lat": 48.8606, "pickup_lon": 2.3376},
        ]
        drivers = [
            {"id": 1, "lat": 48.8566, "lon": 2.3522, "available": True},
            {"id": 2, "lat": 48.8606, "lon": 2.3376, "available": True},
        ]

        suggestions = generator._get_heuristic_suggestions(assignments=assignments, drivers=drivers, max_suggestions=5)

        assert isinstance(suggestions, list)

    def test_calculate_distance(self):
        """Test calcul distance."""
        generator = RLSuggestionGenerator()

        distance = generator._calculate_distance(lat1=48.8566, lon1=2.3522, lat2=48.8606, lon2=2.3376)

        assert isinstance(distance, float)
        assert distance >= 0

    def test_is_model_loaded(self):
        """Test vérification modèle chargé."""
        generator = RLSuggestionGenerator()

        # Test avec agent
        generator.agent = Mock()
        assert generator._is_model_loaded() is True

        # Test sans agent
        generator.agent = None
        assert generator._is_model_loaded() is False

    def test_get_model_info(self):
        """Test informations modèle."""
        generator = RLSuggestionGenerator()

        info = generator.get_model_info()

        assert isinstance(info, dict)
        assert "model_path" in info
        assert "loaded" in info

    def test_reload_model(self):
        """Test rechargement modèle."""
        generator = RLSuggestionGenerator()

        with patch.object(generator, "_load_model") as mock_load:
            generator.reload_model()
            mock_load.assert_called_once()

    def test_clear_model(self):
        """Test nettoyage modèle."""
        generator = RLSuggestionGenerator()
        generator.agent = Mock()
        generator.env = Mock()

        generator.clear_model()

        assert generator.agent is None
        assert generator.env is None
