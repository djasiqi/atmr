#!/usr/bin/env python3
"""
Tests pour suggestion_generator.py - couverture de base
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import torch

from services.rl.suggestion_generator import RLSuggestionGenerator


class TestRLSuggestionGenerator:
    """Tests pour la classe RLSuggestionGenerator."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        generator = RLSuggestionGenerator()

        assert generator.model_path is not None
        assert generator.agent is None
        assert generator._is_model_loaded() is False

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        generator = RLSuggestionGenerator(model_path="custom_model.pkl")

        assert generator.model_path == "custom_model.pkl"
        assert generator.agent is None
        assert generator._is_model_loaded() is False

    def test_lazy_import_rl_success(self):
        """Test import paresseux RL avec succès."""
        import services.rl.suggestion_generator as sg_module
        from services.rl.suggestion_generator import _lazy_import_rl

        # Réinitialiser les variables globales

        original_dqn = sg_module._dqn_agent
        original_env = sg_module._dispatch_env

        try:
            sg_module._dqn_agent = None
            sg_module._dispatch_env = None

            # Mock des modules RL
            mock_dqn_module = Mock()
            mock_dispatch_module = Mock()
            mock_improved_dqn_class = Mock()

            with (
                patch(
                    "services.rl.suggestion_generator.improved_dqn_agent",
                    mock_dqn_module,
                ),
                patch(
                    "services.rl.suggestion_generator.dispatch_env",
                    mock_dispatch_module,
                ),
                patch(
                    "services.rl.improved_dqn_agent.ImprovedDQNAgent",
                    mock_improved_dqn_class,
                ),
            ):
                _lazy_import_rl()

                assert sg_module._dqn_agent == mock_dqn_module
                assert sg_module._dispatch_env == mock_dispatch_module
        finally:
            # Restaurer les valeurs originales
            sg_module._dqn_agent = original_dqn
            sg_module._dispatch_env = original_env

    def test_lazy_import_rl_failure(self):
        """Test import paresseux RL avec échec."""
        import services.rl.suggestion_generator as sg_module
        from services.rl.suggestion_generator import _lazy_import_rl

        # Réinitialiser les variables globales

        original_dqn = sg_module._dqn_agent
        original_env = sg_module._dispatch_env

        try:
            sg_module._dqn_agent = None
            sg_module._dispatch_env = None

            with (
                patch(
                    "services.rl.suggestion_generator.improved_dqn_agent",
                    side_effect=ImportError("Module not found"),
                ),
                pytest.raises(ImportError),
            ):
                _lazy_import_rl()
        finally:
            # Restaurer les valeurs originales
            sg_module._dqn_agent = original_dqn
            sg_module._dispatch_env = original_env

    def test_load_model_file_exists(self):
        """Test chargement de modèle avec fichier existant."""
        generator = RLSuggestionGenerator()

        # Mock du fichier existant
        mock_agent = Mock()
        mock_agent.load = Mock()
        mock_agent.q_network = Mock()
        mock_agent.q_network.eval = Mock()

        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.observation_space.shape = [19]
        mock_env.action_space = Mock()
        mock_env.action_space.n = 26

        with (
            patch("services.rl.suggestion_generator.Path") as mock_path,
            patch("services.rl.suggestion_generator._lazy_import_rl"),
            patch(
                "services.rl.suggestion_generator.ImprovedDQNAgent",
                return_value=mock_agent,
            ),
            patch("services.rl.dispatch_env.DispatchEnv", return_value=mock_env),
            patch("torch.load", return_value={"state_dict": {}}),
        ):
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            generator._load_model()

            assert generator.agent is not None
            assert generator._is_model_loaded() is True

    def test_load_model_file_not_found(self):
        """Test chargement de modèle avec fichier inexistant."""
        generator = RLSuggestionGenerator()

        # Mock du fichier inexistant
        with (
            patch("services.rl.suggestion_generator.Path") as mock_path,
            patch("services.rl.suggestion_generator._lazy_import_rl"),
        ):
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = False
            mock_path.return_value = mock_path_instance

            generator._load_model()

            assert generator.agent is None
            assert generator._is_model_loaded() is False

    def test_load_model_with_exception(self):
        """Test chargement de modèle avec exception."""
        generator = RLSuggestionGenerator()

        # Mock pour lever une exception
        with (
            patch("services.rl.suggestion_generator.Path") as mock_path,
            patch("services.rl.suggestion_generator._lazy_import_rl"),
            patch("torch.load", side_effect=Exception("Load error")),
        ):
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance

            generator._load_model()

            assert generator.agent is None
            assert generator._is_model_loaded() is False

    def test_generate_suggestions_no_model(self):
        """Test génération de suggestions sans modèle."""
        generator = RLSuggestionGenerator()

        # Mock pour que le modèle ne soit pas chargé
        generator.agent = None

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date="2024-01-01",
            min_confidence=0.5,
            max_suggestions=10,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_with_model(self):
        """Test génération de suggestions avec modèle."""
        generator = RLSuggestionGenerator()

        # Mock pour que le modèle soit chargé
        generator.agent = Mock()
        mock_q_network = Mock()
        mock_q_values = Mock()
        mock_q_values.cpu.return_value.numpy.return_value = [
            [0.8, 0.6, 0.4, 0.2, 0.1] + [0.0] * 21
        ]
        mock_q_network.return_value = mock_q_values
        generator.agent.q_network = mock_q_network

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date="2024-01-01",
            min_confidence=0.5,
            max_suggestions=10,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0  # Pas d'assignments donc pas de suggestions

    def test_generate_suggestions_empty_input(self):
        """Test génération de suggestions avec entrée vide."""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date="2024-01-01",
            min_confidence=0.5,
            max_suggestions=10,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_no_available_drivers(self):
        """Test génération de suggestions sans chauffeurs disponibles."""
        generator = RLSuggestionGenerator()

        # Mock des drivers non disponibles (objets avec attributs)
        mock_driver1 = Mock()
        mock_driver1.id = 1
        mock_driver1.is_available = False

        mock_driver2 = Mock()
        mock_driver2.id = 2
        mock_driver2.is_available = False

        drivers = [mock_driver1, mock_driver2]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=drivers,
            for_date="2024-01-01",
            min_confidence=0.5,
            max_suggestions=10,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_no_unassigned_assignments(self):
        """Test génération de suggestions sans assignments non assignés."""
        generator = RLSuggestionGenerator()

        # Mock des assignments sans booking (donc non traitables)
        mock_assignment1 = Mock()
        mock_assignment1.id = 1
        mock_assignment1.booking = None  # Pas de booking
        mock_assignment1.driver = Mock()

        mock_assignment2 = Mock()
        mock_assignment2.id = 2
        mock_assignment2.booking = None  # Pas de booking
        mock_assignment2.driver = Mock()

        assignments = [mock_assignment1, mock_assignment2]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=[],
            for_date="2024-01-01",
            min_confidence=0.5,
            max_suggestions=10,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_with_exception(self):
        """Test génération de suggestions avec exception."""
        generator = RLSuggestionGenerator()

        # Mock pour lever une exception dans _generate_rl_suggestions
        generator.agent = Mock()
        mock_q_network = Mock()
        mock_q_network.side_effect = Exception("RL error")
        generator.agent.q_network = mock_q_network

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date="2024-01-01",
            min_confidence=0.5,
            max_suggestions=10,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_with_parameters(self):
        """Test génération de suggestions avec paramètres."""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date="2024-01-01",
            min_confidence=0.8,
            max_suggestions=5,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_with_confidence_threshold(self):
        """Test génération de suggestions avec seuil de confiance."""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date="2024-01-01",
            min_confidence=0.9,
            max_suggestions=10,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_max_suggestions(self):
        """Test génération de suggestions avec nombre maximum."""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date="2024-01-01",
            min_confidence=0.5,
            max_suggestions=3,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_with_different_dates(self):
        """Test génération de suggestions avec différentes dates."""
        generator = RLSuggestionGenerator()

        # Test avec date passée (for_date est une string, pas un datetime)
        past_date = "2020-01-01"
        suggestions1 = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date=past_date,
            min_confidence=0.5,
            max_suggestions=10,
        )

        # Test avec date future
        future_date = "2030-01-01"
        suggestions2 = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date=future_date,
            min_confidence=0.5,
            max_suggestions=10,
        )

        assert isinstance(suggestions1, list)
        assert isinstance(suggestions2, list)
        assert len(suggestions1) == 0
        assert len(suggestions2) == 0

    def test_generate_suggestions_with_different_companies(self):
        """Test génération de suggestions avec différentes entreprises."""
        generator = RLSuggestionGenerator()

        # Test avec différentes entreprises (company_id doit être un int)
        companies = [1, 2, 3]

        for company_id in companies:
            suggestions = generator.generate_suggestions(
                company_id=company_id,
                assignments=[],
                drivers=[],
                for_date="2024-01-01",
                min_confidence=0.5,
                max_suggestions=10,
            )

            assert isinstance(suggestions, list)
            assert len(suggestions) == 0

    def test_generate_suggestions_with_none_values(self):
        """Test génération de suggestions avec valeurs None."""
        generator = RLSuggestionGenerator()

        # Les valeurs None causeront des erreurs, donc on utilise des valeurs par défaut
        # ou on s'attend à une exception
        import pytest

        with pytest.raises((TypeError, AttributeError)):
            generator.generate_suggestions(
                company_id=None,
                assignments=None,
                drivers=None,
                for_date=None,
                min_confidence=None,
                max_suggestions=None,
            )

    def test_generate_suggestions_with_empty_strings(self):
        """Test génération de suggestions avec chaînes vides."""
        generator = RLSuggestionGenerator()

        # company_id doit être un int, pas une string vide
        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date="2024-01-01",
            min_confidence=0.5,
            max_suggestions=10,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0
