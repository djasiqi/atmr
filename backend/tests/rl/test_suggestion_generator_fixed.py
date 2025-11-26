#!/usr/bin/env python3
"""
Tests simplifiés pour suggestion_generator.py - avec objets mock appropriés
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from services.rl.suggestion_generator import RLSuggestionGenerator, _lazy_import_rl


class TestLazyImport:
    """Tests pour la fonction _lazy_import_rl."""

    def test_lazy_import_success(self):
        """Test import réussi des modules RL."""
        import services.rl.suggestion_generator as sg_module

        original_dqn = sg_module._dqn_agent
        original_env = sg_module._dispatch_env

        try:
            sg_module._dqn_agent = None
            sg_module._dispatch_env = None

            mock_dqn_module = Mock()
            mock_dispatch_module = Mock()
            mock_improved_dqn_class = Mock()

            with (
                patch(
                    "services.rl.improved_dqn_agent",
                    mock_dqn_module,
                ),
                patch(
                    "services.rl.dispatch_env",
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
            sg_module._dqn_agent = original_dqn
            sg_module._dispatch_env = original_env

    def test_lazy_import_failure(self):
        """Test échec d'import des modules RL."""
        import services.rl.suggestion_generator as sg_module
        from services.rl.suggestion_generator import _lazy_import_rl

        original_dqn = sg_module._dqn_agent
        original_env = sg_module._dispatch_env

        try:
            sg_module._dqn_agent = None
            sg_module._dispatch_env = None

            # ✅ FIX: Supprimer le module du cache et utiliser importlib pour intercepter
            import importlib
            import sys

            # Sauvegarder et supprimer le module du cache pour forcer un nouvel import
            original_module = sys.modules.pop("services.rl.improved_dqn_agent", None)
            try:
                # Créer une fonction qui lève ImportError pour ce module spécifique
                original_import_module = importlib.import_module

                def failing_import_module(name, package=None):
                    if name == "services.rl.improved_dqn_agent":
                        raise ImportError("Module not found")
                    return original_import_module(name, package)

                with (
                    patch("importlib.import_module", side_effect=failing_import_module),
                    pytest.raises(ImportError),
                ):
                    _lazy_import_rl()
            finally:
                # Restaurer le module original
                if original_module is not None:
                    sys.modules["services.rl.improved_dqn_agent"] = original_module
                elif "services.rl.improved_dqn_agent" in sys.modules:
                    del sys.modules["services.rl.improved_dqn_agent"]
        finally:
            sg_module._dqn_agent = original_dqn
            sg_module._dispatch_env = original_env


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

    @patch("pathlib.Path")
    def test_load_model_file_not_exists(self, mock_path_class):
        """Test chargement de modèle quand le fichier n'existe pas."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path_class.return_value = mock_path_instance

        generator = RLSuggestionGenerator()

        # Le modèle ne devrait pas être chargé
        assert generator.agent is None
        assert generator.env is None

    @patch("pathlib.Path")
    @patch("services.rl.suggestion_generator._lazy_import_rl")
    def test_load_model_file_exists(
        self,
        mock_lazy_import,
        mock_path_class,
    ):
        """Test chargement de modèle quand le fichier existe."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance

        # Mock de l'agent et de l'environnement
        mock_agent = Mock()
        mock_agent.load = Mock()
        mock_agent.q_network = Mock()
        mock_agent.q_network.eval = Mock()

        mock_env = Mock()
        mock_env.observation_space = Mock()
        mock_env.observation_space.shape = [19]
        mock_env.action_space = Mock()
        mock_env.action_space.n = 26

        # ✅ FIX: Patcher les classes à la source (services.rl) plutôt que
        # dans suggestion_generator car les imports sont faits dans _load_model()
        # ✅ FIX: Réinitialiser _model_loaded et créer le générateur dans le bloc with
        import services.rl.suggestion_generator as sg_module

        original_model_loaded = sg_module._model_loaded
        try:
            sg_module._model_loaded = False

            with (
                patch(
                    "services.rl.improved_dqn_agent.ImprovedDQNAgent",
                    return_value=mock_agent,
                ),
                patch("services.rl.dispatch_env.DispatchEnv", return_value=mock_env),
                patch(
                    "torch.load",
                    return_value={
                        "q_network_state_dict": {},
                        "target_network_state_dict": {},
                        "optimizer_state_dict": {},
                        "epsilon": 0.1,
                        "training_step": 0,
                        "episode_count": 0,
                        "losses": [],
                    },
                ),
            ):
                # Créer le générateur après avoir mis en place les patches
                generator = RLSuggestionGenerator()

                mock_lazy_import.assert_called_once()
                assert generator.agent == mock_agent
        finally:
            sg_module._model_loaded = original_model_loaded

    @patch("pathlib.Path")
    @patch("services.rl.suggestion_generator._lazy_import_rl")
    def test_load_model_torch_load_error(
        self,
        mock_lazy_import,
        mock_path_class,
    ):
        """Test chargement de modèle avec erreur torch.load."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path_class.return_value = mock_path_instance

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
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 46.2
        mock_booking.pickup_lon = 6.1
        mock_booking.dropoff_lat = 46.3
        mock_booking.dropoff_lon = 6.2
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 46.2
        mock_driver.current_lon = 6.1
        mock_driver.driver_type = Mock()
        mock_driver.driver_type.value = "REGULAR"
        mock_driver.user = None

        mock_assignment = Mock()
        mock_assignment.id = 1
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [mock_driver]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-01-01",
        )

        # Devrait retourner des suggestions basiques
        assert isinstance(suggestions, list)

    def test_generate_suggestions_empty_input(self):
        """Test génération de suggestions avec entrée vide."""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=[], drivers=[], for_date="2024-01-01"
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_with_model(self):
        """Test génération de suggestions avec modèle."""
        generator = RLSuggestionGenerator()

        # Mock de l'agent
        mock_agent = Mock()
        mock_q_network = Mock()
        mock_q_values = Mock()
        mock_q_values.cpu.return_value.numpy.return_value = [
            [0.8, 0.6, 0.4, 0.2, 0.1] + [0.0] * 21
        ]
        mock_q_network.return_value = mock_q_values
        generator.agent = mock_agent
        generator.agent.q_network = mock_q_network

        # Créer des objets mock appropriés
        from datetime import datetime, timedelta

        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 46.2
        mock_booking.pickup_lon = 6.1
        mock_booking.dropoff_lat = 46.3
        mock_booking.dropoff_lon = 6.2
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 46.2
        mock_driver.current_lon = 6.1
        mock_driver.driver_type = Mock()
        mock_driver.driver_type.value = "REGULAR"
        mock_driver.user = None

        mock_assignment = Mock()
        mock_assignment.id = 1
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [mock_driver]

        # ✅ FIX: Patcher models.Assignment avant l'import dans generate_suggestions
        # pour éviter l'erreur de contexte Flask
        mock_assignment_model = Mock()
        mock_query = Mock()
        mock_query.filter.return_value.count.return_value = 0
        mock_assignment_model.query = mock_query

        with patch("models.Assignment", mock_assignment_model):
            suggestions = generator.generate_suggestions(
                company_id=1,
                assignments=assignments,
                drivers=drivers,
                for_date="2024-01-01",
            )

            assert isinstance(suggestions, list)

    def test_generate_suggestions_with_exception(self):
        """Test génération de suggestions avec exception."""
        generator = RLSuggestionGenerator()

        # Mock de l'agent qui lève une exception
        mock_agent = Mock()
        mock_agent.q_network.side_effect = Exception("Model error")

        generator.agent = mock_agent

        # Créer des objets mock appropriés
        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 46.2
        mock_booking.pickup_lon = 6.1
        mock_booking.dropoff_lat = 46.3
        mock_booking.dropoff_lon = 6.2
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 46.2
        mock_driver.current_lon = 6.1
        mock_driver.driver_type = Mock()
        mock_driver.driver_type.value = "REGULAR"
        mock_driver.user = None

        mock_assignment = Mock()
        mock_assignment.id = 1
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [mock_driver]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-01-01",
        )

        # Devrait retourner des suggestions basiques en cas d'erreur
        assert isinstance(suggestions, list)

    def test_generate_suggestions_with_parameters(self):
        """Test génération de suggestions avec paramètres."""
        generator = RLSuggestionGenerator()

        # Créer des objets mock appropriés
        assignments = []
        for i in range(5):
            mock_booking = Mock()
            mock_booking.id = i
            mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
            mock_booking.pickup_lat = 46.2 + i * 0.1
            mock_booking.pickup_lon = 6.1 + i * 0.1
            mock_booking.dropoff_lat = 46.3 + i * 0.1
            mock_booking.dropoff_lon = 6.2 + i * 0.1
            mock_booking.is_emergency = False

            mock_driver = Mock()
            mock_driver.id = i
            mock_driver.is_available = True
            mock_driver.current_lat = 46.2 + i * 0.1
            mock_driver.current_lon = 6.1 + i * 0.1
            mock_driver.driver_type = Mock()
            mock_driver.driver_type.value = "REGULAR"
            mock_driver.user = None

            mock_assignment = Mock()
            mock_assignment.id = i
            mock_assignment.booking = mock_booking
            mock_assignment.driver = mock_driver
            assignments.append(mock_assignment)

        # Convertir les dictionnaires en objets mock
        drivers = []
        for i in range(5):
            mock_driver_obj = Mock()
            mock_driver_obj.id = i
            mock_driver_obj.is_available = True
            mock_driver_obj.current_lat = 46.2 + i * 0.1
            mock_driver_obj.current_lon = 6.1 + i * 0.1
            mock_driver_obj.driver_type = Mock()
            mock_driver_obj.driver_type.value = "REGULAR"
            mock_driver_obj.user = None
            drivers.append(mock_driver_obj)

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-01-01",
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
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 46.2
        mock_booking.pickup_lon = 6.1
        mock_booking.dropoff_lat = 46.3
        mock_booking.dropoff_lon = 6.2
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = False  # Non disponible
        mock_driver.current_lat = 46.2
        mock_driver.current_lon = 6.1
        mock_driver.driver_type = Mock()
        mock_driver.driver_type.value = "REGULAR"
        mock_driver.user = None

        mock_assignment = Mock()
        mock_assignment.id = 1
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [mock_driver]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-01-01",
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_generate_suggestions_no_unassigned_assignments(self):
        """Test génération de suggestions sans assignments non assignés."""
        generator = RLSuggestionGenerator()

        # Créer des objets mock appropriés (sans booking pour simuler
        # un assignment non assignable)
        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 46.2
        mock_driver.current_lon = 6.1
        mock_driver.driver_type = Mock()
        mock_driver.driver_type.value = "REGULAR"
        mock_driver.user = None

        mock_assignment = Mock()
        mock_assignment.id = 1
        mock_assignment.booking = None  # Pas de booking = non assignable
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [mock_driver]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-01-01",
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0
