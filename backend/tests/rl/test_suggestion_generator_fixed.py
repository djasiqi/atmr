#!/usr/bin/env python3
"""
Tests simplifiés pour suggestion_generator.py - avec objets mock appropriés
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
        # Mock des modules globaux
        with (
            patch("services.rl.suggestion_generator._dqn_agent", None),
            patch("services.rl.suggestion_generator._dispatch_env", None),
        ):
            # Mock des imports
            mock_dqn_agent = Mock()
            mock_dispatch_env = Mock()

            with patch("importlib.import_module") as mock_import:
                mock_import.side_effect = [Mock(ImprovedDQNAgent=mock_dqn_agent), Mock(DispatchEnv=mock_dispatch_env)]

                _lazy_import_rl()

                from services.rl.suggestion_generator import _dispatch_env, _dqn_agent

                assert _dqn_agent == mock_dqn_agent
                assert _dispatch_env == mock_dispatch_env

    def test_lazy_import_failure(self):
        """Test échec d'import des modules RL."""
        with (
            patch("services.rl.suggestion_generator._dqn_agent", None),
            patch("services.rl.suggestion_generator._dispatch_env", None),
            patch("importlib.import_module", side_effect=ImportError("Module not found")),
            pytest.raises(ImportError),
        ):
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
    def test_load_model_file_exists(
        self,
        mock_lazy_import,
        mock_path,
    ):
        """Test chargement de modèle quand le fichier existe."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance

        # Mock de l'agent et de l'environnement
        mock_agent = Mock()
        mock_env = Mock()

        with (
            patch("services.rl.suggestion_generator._dqn_agent") as mock_dqn_module,
            patch("services.rl.suggestion_generator._dispatch_env") as mock_env_module,
            patch("torch.load", return_value={"state_dict": {}}),
        ):
            mock_dqn_module.ImprovedDQNAgent.return_value = mock_agent
            mock_env_module.DispatchEnv.return_value = mock_env

            generator = RLSuggestionGenerator()

            mock_lazy_import.assert_called_once()
            assert generator.agent == mock_agent
            assert generator.env == mock_env

    @patch("services.rl.suggestion_generator.Path")
    @patch("services.rl.suggestion_generator._lazy_import_rl")
    def test_load_model_torch_load_error(
        self,
        mock_lazy_import,
        mock_path,
    ):
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

        # Créer des objets mock appropriés
        mock_booking = Mock()
        mock_driver = Mock()
        mock_assignment = Mock()
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": True}]

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=assignments, drivers=drivers, for_date="2024-0.1-0.1"
        )

        # Devrait retourner des suggestions basiques
        assert isinstance(suggestions, list)
        assert len(suggestions) == 0  # Pas de suggestions sans modèle

    def test_generate_suggestions_empty_input(self):
        """Test génération de suggestions avec entrée vide."""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(company_id=1, assignments=[], drivers=[], for_date="2024-0.1-0.1")

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

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

        # Créer des objets mock appropriés
        mock_booking = Mock()
        mock_driver = Mock()
        mock_assignment = Mock()
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": True}]

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=assignments, drivers=drivers, for_date="2024-0.1-0.1"
        )

        assert isinstance(suggestions, list)
        mock_env.get_state.assert_called_once()

    def test_generate_suggestions_with_exception(self):
        """Test génération de suggestions avec exception."""
        generator = RLSuggestionGenerator()

        # Mock de l'agent qui lève une exception
        mock_agent = Mock()
        mock_agent.q_network.side_effect = Exception("Model error")

        generator.agent = mock_agent

        # Créer des objets mock appropriés
        mock_booking = Mock()
        mock_driver = Mock()
        mock_assignment = Mock()
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": True}]

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=assignments, drivers=drivers, for_date="2024-0.1-0.1"
        )

        # Devrait retourner des suggestions basiques en cas d'erreur
        assert isinstance(suggestions, list)

    def test_generate_suggestions_with_parameters(self):
        """Test génération de suggestions avec paramètres."""
        generator = RLSuggestionGenerator()

        # Créer des objets mock appropriés
        assignments = []
        for _i in range(5):
            mock_booking = Mock()
            mock_driver = Mock()
            mock_assignment = Mock()
            mock_assignment.booking = mock_booking
            mock_assignment.driver = mock_driver
            assignments.append(mock_assignment)

        drivers = [{"id": i, "lat": 46.2 + i * 0.1, "lon": 6.1 + i * 0.1, "available": True} for i in range(5)]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-0.1-0.1",
            min_confidence=0.7,
            max_suggestions=3,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3

    def test_generate_suggestions_no_available_drivers(self):
        """Test génération de suggestions sans chauffeurs disponibles."""
        generator = RLSuggestionGenerator()

        # Créer des objets mock appropriés
        mock_booking = Mock()
        mock_driver = Mock()
        mock_assignment = Mock()
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": False}]

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=assignments, drivers=drivers, for_date="2024-0.1-0.1"
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_no_unassigned_assignments(self):
        """Test génération de suggestions sans assignments non assignés."""
        generator = RLSuggestionGenerator()

        # Créer des objets mock appropriés
        mock_booking = Mock()
        mock_driver = Mock()
        mock_assignment = Mock()
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [{"id": 1, "lat": 46.2, "lon": 6.1, "available": True}]

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=assignments, drivers=drivers, for_date="2024-0.1-0.1"
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0
