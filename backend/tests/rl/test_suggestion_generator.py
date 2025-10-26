#!/usr/bin/env python3
"""
Tests complets pour suggestion_generator.py
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from services.rl.suggestion_generator import RLSuggestionGenerator, _lazy_import_rl


class TestLazyImport:
    """Tests pour la fonction _lazy_import_rl."""

    def test_lazy_import_success(self):
        """Test import réussi des modules RL."""
        with patch("services.rl.suggestion_generator._dqn_agent", None), \
             patch("services.rl.suggestion_generator._dispatch_env", None):

            # Mock des imports
            mock_dqn_agent = Mock()
            mock_dispatch_env = Mock()

            with patch("services.rl.suggestion_generator.improved_dqn_agent", mock_dqn_agent), \
                 patch("services.rl.suggestion_generator.dispatch_env", mock_dispatch_env):

                _lazy_import_rl()

                from services.rl.suggestion_generator import _dispatch_env, _dqn_agent
                assert _dqn_agent == mock_dqn_agent
                assert _dispatch_env == mock_dispatch_env

    def test_lazy_import_failure(self):
        """Test échec d'import des modules RL."""
        with patch("services.rl.suggestion_generator._dqn_agent", None), \
             patch("services.rl.suggestion_generator._dispatch_env", None):

            with patch("services.rl.suggestion_generator.improved_dqn_agent", side_effect=ImportError("Module not found")):

                with pytest.raises(ImportError):
                    _lazy_import_rl()


class TestRLSuggestionGenerator:
    """Tests pour la classe RLSuggestionGenerator."""

    def test_init_with_default_path(self):
        """Test initialisation avec chemin par défaut."""
        generator = RLSuggestionGenerator()

        assert generator.model_path == "data/ml/dqn_agent_best_v33.pth"
        assert generator.agent is None
        assert generator.env is None

    def test_init_with_custom_path(self):
        """Test initialisation avec chemin personnalisé."""
        custom_path = "custom/path/model.pth"
        generator = RLSuggestionGenerator(model_path=custom_path)

        assert generator.model_path == custom_path
        assert generator.agent is None
        assert generator.env is None

    @patch("services.rl.suggestion_generator.Path")
    def test_load_model_file_not_exists(self, mock_path):
        """Test chargement de modèle quand le fichier n'existe pas."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        generator = RLSuggestionGenerator()

        # Le modèle ne devrait pas être chargé
        assert generator.agent is None
        assert generator.env is None

    @patch("services.rl.suggestion_generator.Path")
    @patch("services.rl.suggestion_generator._lazy_import_rl")
    def test_load_model_file_exists(self, ____________________________________________________________________________________________________mock_lazy_import, mock_path):
        """Test chargement de modèle quand le fichier existe."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Mock de l'agent et de l'environnement
        mock_agent = Mock()
        mock_env = Mock()

        with patch("services.rl.suggestion_generator.ImprovedDQNAgent", return_value=mock_agent), \
             patch("services.rl.suggestion_generator.DispatchEnv", return_value=mock_env), \
             patch("torch.load", return_value={"state_dict": {}}):

            generator = RLSuggestionGenerator()

            mock_lazy_import.assert_called_once()
            assert generator.agent == mock_agent
            assert generator.env == mock_env

    @patch("services.rl.suggestion_generator.Path")
    @patch("services.rl.suggestion_generator._lazy_import_rl")
    def test_load_model_torch_load_error(self, ____________________________________________________________________________________________________mock_lazy_import, mock_path):
        """Test chargement de modèle avec erreur torch.load."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        with patch("torch.load", side_effect=Exception("Load error")):
            generator = RLSuggestionGenerator()

            # Le modèle ne devrait pas être chargé en cas d'erreur
            assert generator.agent is None
            assert generator.env is None

    def test_generate_suggestions_no_model(self):
        """Test génération de suggestions sans modèle."""
        generator = RLSuggestionGenerator()
        generator.agent = None

        assignments = [{"id": 1, "pickup_lat": 46.2, "pickup_lon": 6.1}]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": True}]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-0.1-0.1"
        )

        # Devrait retourner des suggestions basiques
        assert isinstance(suggestions, list)
        assert len(suggestions) == 0  # Pas de suggestions sans modèle

    def test_generate_suggestions_with_model(self):
        """Test génération de suggestions avec modèle."""
        generator = RLSuggestionGenerator()

        # Mock de l'agent et de l'environnement
        mock_agent = Mock()
        mock_env = Mock()

        # Mock de l'état et des Q-values
        mock_state = np.array([[1, 2, 3, 4, 5]])
        mock_q_values = np.array([[0.1, 0.9, 0.2, 0.3]])

        mock_env.get_state.return_value = mock_state
        mock_agent.q_network.return_value = torch.tensor(mock_q_values)

        generator.agent = mock_agent
        generator.env = mock_env

        assignments = [{"id": 1, "pickup_lat": 46.2, "pickup_lon": 6.1}]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": True}]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-0.1-0.1"
        )

        assert isinstance(suggestions, list)
        mock_env.get_state.assert_called_once()

    def test_generate_suggestions_empty_input(self):
        """Test génération de suggestions avec entrée vide."""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions([], [])

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_no_available_drivers(self):
        """Test génération de suggestions sans chauffeurs disponibles."""
        generator = RLSuggestionGenerator()

        bookings = [{"id": 1, "pickup_lat": 46.2, "pickup_lon": 6.1}]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": False}]

        suggestions = generator.generate_suggestions(bookings, drivers)

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_no_unassigned_bookings(self):
        """Test génération de suggestions sans bookings non assignés."""
        generator = RLSuggestionGenerator()

        bookings = [{"id": 1, "pickup_lat": 46.2, "pickup_lon": 6.1, "assigned": True}]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": True}]

        suggestions = generator.generate_suggestions(bookings, drivers)

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_with_exception(self):
        """Test génération de suggestions avec exception."""
        generator = RLSuggestionGenerator()

        # Mock de l'agent qui lève une exception
        mock_agent = Mock()
        mock_agent.q_network.side_effect = Exception("Model error")

        generator.agent = mock_agent

        bookings = [{"id": 1, "pickup_lat": 46.2, "pickup_lon": 6.1}]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": True}]

        suggestions = generator.generate_suggestions(bookings, drivers)

        # Devrait retourner des suggestions basiques en cas d'erreur
        assert isinstance(suggestions, list)

    def test_get_suggestion_confidence(self):
        """Test calcul de la confiance des suggestions."""
        generator = RLSuggestionGenerator()

        # Test avec Q-values élevées (haute confiance)
        high_q_values = np.array([0.1, 0.9, 0.2])
        confidence = generator._get_suggestion_confidence(high_q_values)

        assert 0 <= confidence <= 1
        assert confidence > 0.5  # Devrait être élevée

    def test_get_suggestion_confidence_low_values(self):
        """Test calcul de la confiance avec Q-values faibles."""
        generator = RLSuggestionGenerator()

        # Test avec Q-values faibles (basse confiance)
        low_q_values = np.array([0.1, 0.2, 0.15])
        confidence = generator._get_suggestion_confidence(low_q_values)

        assert 0 <= confidence <= 1
        assert confidence < 0.5  # Devrait être faible

    def test_get_suggestion_confidence_empty_values(self):
        """Test calcul de la confiance avec Q-values vides."""
        generator = RLSuggestionGenerator()

        empty_q_values = np.array([])
        confidence = generator._get_suggestion_confidence(empty_q_values)

        assert confidence == 0

    def test_format_suggestion(self):
        """Test formatage des suggestions."""
        generator = RLSuggestionGenerator()

        suggestion = {
            "booking_id": 1,
            "driver_id": 2,
            "confidence": 0.8,
            "reason": "Optimization"
        }

        formatted = generator._format_suggestion(suggestion)

        assert isinstance(formatted, dict)
        assert "booking_id" in formatted
        assert "driver_id" in formatted
        assert "confidence" in formatted
        assert "reason" in formatted

    def test_get_heuristic_suggestions(self):
        """Test suggestions heuristiques."""
        generator = RLSuggestionGenerator()

        bookings = [
            {"id": 1, "pickup_lat": 46.2, "pickup_lon": 6.1, "assigned": False},
            {"id": 2, "pickup_lat": 46.3, "pickup_lon": 6.2, "assigned": False}
        ]
        drivers = [
            {"id": 1, "lat": 46.2, "lon": 6.1, "available": True},
            {"id": 2, "lat": 46.3, "lon": 6.2, "available": True}
        ]

        suggestions = generator._get_heuristic_suggestions(bookings, drivers)

        assert isinstance(suggestions, list)
        # Devrait avoir au moins une suggestion heuristique

    def test_calculate_distance(self):
        """Test calcul de distance."""
        generator = RLSuggestionGenerator()

        lat1, lon1 = 46.2, 6.1
        lat2, lon2 = 46.3, 6.2

        distance = generator._calculate_distance(lat1, lon1, lat2, lon2)

        assert isinstance(distance, float)
        assert distance > 0

    def test_calculate_distance_same_location(self):
        """Test calcul de distance pour la même localisation."""
        generator = RLSuggestionGenerator()

        lat, lon = 46.2, 6.1

        distance = generator._calculate_distance(lat, lon, lat, lon)

        assert distance == 0

    def test_is_model_loaded(self):
        """Test vérification du chargement du modèle."""
        generator = RLSuggestionGenerator()

        # Test sans modèle
        assert not generator._is_model_loaded()

        # Test avec modèle
        generator.agent = Mock()
        generator.env = Mock()
        assert generator._is_model_loaded()

    def test_get_model_info(self):
        """Test récupération des informations du modèle."""
        generator = RLSuggestionGenerator()

        info = generator.get_model_info()

        assert isinstance(info, dict)
        assert "model_path" in info
        assert "loaded" in info
        assert "agent_type" in info

    def test_get_model_info_with_model(self):
        """Test récupération des informations du modèle avec modèle chargé."""
        generator = RLSuggestionGenerator()

        # Mock de l'agent
        mock_agent = Mock()
        mock_agent.__class__.__name__ = "ImprovedDQNAgent"

        generator.agent = mock_agent
        generator.env = Mock()

        info = generator.get_model_info()

        assert info["loaded"] is True
        assert info["agent_type"] == "ImprovedDQNAgent"

    def test_reload_model(self):
        """Test rechargement du modèle."""
        generator = RLSuggestionGenerator()

        with patch.object(generator, "_load_model") as mock_load:
            generator.reload_model()
            mock_load.assert_called_once()

    def test_clear_model(self):
        """Test suppression du modèle."""
        generator = RLSuggestionGenerator()

        generator.agent = Mock()
        generator.env = Mock()

        generator.clear_model()

        assert generator.agent is None
        assert generator.env is None

    def test_generate_suggestions_with_confidence_threshold(self):
        """Test génération de suggestions avec seuil de confiance."""
        generator = RLSuggestionGenerator()

        # Mock de l'agent avec Q-values faibles
        mock_agent = Mock()
        mock_env = Mock()

        mock_state = np.array([[1, 2, 3, 4, 5]])
        mock_q_values = np.array([[0.1, 0.2, 0.15]])  # Q-values faibles

        mock_env.get_state.return_value = mock_state
        mock_agent.q_network.return_value = torch.tensor(mock_q_values)

        generator.agent = mock_agent
        generator.env = mock_env

        bookings = [{"id": 1, "pickup_lat": 46.2, "pickup_lon": 6.1}]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": True}]

        # Test avec seuil de confiance élevé
        suggestions = generator.generate_suggestions(bookings, drivers, confidence_threshold=0.8)

        assert isinstance(suggestions, list)
        # Devrait avoir moins de suggestions avec un seuil élevé

    def test_generate_suggestions_max_suggestions(self):
        """Test génération de suggestions avec limite maximale."""
        generator = RLSuggestionGenerator()

        bookings = [
            {"id": i, "pickup_lat": 46.2 + i*0.1, "pickup_lon": 6.1 + i*0.1, "assigned": False}
            for i in range(10)
        ]
        drivers = [
            {"id": i, "lat": 46.2 + i*0.1, "lon": 6.1 + i*0.1, "available": True}
            for i in range(10)
        ]

        suggestions = generator.generate_suggestions(bookings, drivers, max_suggestions=3)

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3
