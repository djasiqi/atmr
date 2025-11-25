"""
Tests simplifiés pour RLSuggestionGenerator
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from services.rl.suggestion_generator import RLSuggestionGenerator


class TestRLSuggestionGeneratorSimple:
    """Tests simplifiés pour RLSuggestionGenerator"""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut"""
        with (
            patch("pathlib.Path") as mock_path_class,
            patch("services.rl.suggestion_generator._lazy_import_rl"),
            patch("services.rl.dispatch_env.DispatchEnv") as mock_env_class,
            patch(
                "services.rl.suggestion_generator.ImprovedDQNAgent"
            ) as mock_agent_class,
        ):
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_class.return_value = mock_path_instance

            mock_env = Mock()
            mock_env.observation_space = Mock()
            mock_env.observation_space.shape = [19]
            mock_env.action_space = Mock()
            mock_env.action_space.n = 26
            mock_env_class.return_value = mock_env

            mock_agent = Mock()
            mock_agent.load = Mock()
            mock_agent.q_network = Mock()
            mock_agent.q_network.eval = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Vérifier que l'agent est créé
            assert generator.agent is not None
            assert generator.model_path == "data/ml/dqn_agent_best_v33.pth"

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés"""
        with (
            patch("pathlib.Path") as mock_path_class,
            patch("services.rl.suggestion_generator._lazy_import_rl"),
            patch("services.rl.dispatch_env.DispatchEnv") as mock_env_class,
            patch(
                "services.rl.suggestion_generator.ImprovedDQNAgent"
            ) as mock_agent_class,
        ):
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = True
            mock_path_class.return_value = mock_path_instance

            mock_env = Mock()
            mock_env.observation_space = Mock()
            mock_env.observation_space.shape = [19]
            mock_env.action_space = Mock()
            mock_env.action_space.n = 26
            mock_env_class.return_value = mock_env

            mock_agent = Mock()
            mock_agent.load = Mock()
            mock_agent.q_network = Mock()
            mock_agent.q_network.eval = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator(model_path="custom/path.pth")

            assert generator.model_path == "custom/path.pth"

    def test_generate_suggestions_no_model(self):
        """Test génération de suggestions sans modèle chargé"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des objets mock avec attributs
        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 45.0
        mock_booking.pickup_lon = 2.0
        mock_booking.dropoff_lat = 45.1
        mock_booking.dropoff_lon = 2.1
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 45.0
        mock_driver.current_lon = 2.0
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
        """Test génération de suggestions avec entrée vide"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=[], drivers=[], for_date="2024-01-01"
        )

        assert suggestions == []

    def test_generate_suggestions_no_available_drivers(self):
        """Test génération de suggestions sans chauffeurs disponibles"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des objets mock avec attributs
        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 45.0
        mock_booking.pickup_lon = 2.0
        mock_booking.dropoff_lat = 45.1
        mock_booking.dropoff_lon = 2.1
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = False
        mock_driver.current_lat = 45.0
        mock_driver.current_lon = 2.0
        mock_driver.driver_type = Mock()
        mock_driver.driver_type.value = "REGULAR"
        mock_driver.user = None

        mock_assignment = Mock()
        mock_assignment.id = 1
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=assignments, drivers=[], for_date="2024-01-01"
        )

        assert suggestions == []

    def test_generate_suggestions_with_exception(self):
        """Test génération de suggestions avec exception"""
        import pytest

        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Mock pour provoquer une exception
        with (
            patch.object(
                generator,
                "_generate_basic_suggestions",
                side_effect=Exception("Test error"),
            ),
            pytest.raises(Exception, match="Test error"),
        ):
            # L'exception devrait être propagée
            generator.generate_suggestions(
                company_id=1, assignments=[], drivers=[], for_date="2024-01-01"
            )

    def test_generate_suggestions_with_parameters(self):
        """Test génération de suggestions avec paramètres"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des objets mock avec attributs
        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 45.0
        mock_booking.pickup_lon = 2.0
        mock_booking.dropoff_lat = 45.1
        mock_booking.dropoff_lon = 2.1
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 45.0
        mock_driver.current_lon = 2.0
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
            min_confidence=0.8,
            max_suggestions=5,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5

    def test_generate_suggestions_with_confidence_threshold(self):
        """Test génération de suggestions avec seuil de confiance"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des objets mock avec attributs
        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 45.0
        mock_booking.pickup_lon = 2.0
        mock_booking.dropoff_lat = 45.1
        mock_booking.dropoff_lon = 2.1
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 45.0
        mock_driver.current_lon = 2.0
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
            min_confidence=0.9,
        )

        assert isinstance(suggestions, list)

    def test_generate_suggestions_max_suggestions(self):
        """Test génération de suggestions avec limite maximale"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des objets mock avec attributs
        assignments = []
        drivers = []

        for i in range(10):
            mock_booking = Mock()
            mock_booking.id = i
            mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
            mock_booking.pickup_lat = 45.0 + i * 0.01
            mock_booking.pickup_lon = 2.0 + i * 0.01
            mock_booking.dropoff_lat = 45.1 + i * 0.01
            mock_booking.dropoff_lon = 2.1 + i * 0.01
            mock_booking.is_emergency = False

            mock_driver = Mock()
            mock_driver.id = i
            mock_driver.is_available = True
            mock_driver.current_lat = 45.0 + i * 0.01
            mock_driver.current_lon = 2.0 + i * 0.01
            mock_driver.driver_type = Mock()
            mock_driver.driver_type.value = "REGULAR"
            mock_driver.user = None

            mock_assignment = Mock()
            mock_assignment.id = i
            mock_assignment.booking = mock_booking
            mock_assignment.driver = mock_driver

            assignments.append(mock_assignment)
            drivers.append(mock_driver)

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-01-01",
            max_suggestions=3,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3

    def test_generate_suggestions_with_different_dates(self):
        """Test génération de suggestions avec différentes dates"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des objets mock avec attributs
        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 45.0
        mock_booking.pickup_lon = 2.0
        mock_booking.dropoff_lat = 45.1
        mock_booking.dropoff_lon = 2.1
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 45.0
        mock_driver.current_lon = 2.0
        mock_driver.driver_type = Mock()
        mock_driver.driver_type.value = "REGULAR"
        mock_driver.user = None

        mock_assignment = Mock()
        mock_assignment.id = 1
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [mock_driver]

        # Test avec date d'aujourd'hui (format string)
        suggestions1 = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-01-01",
        )

        # Test avec date future (format string)
        suggestions2 = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-12-31",
        )

        assert isinstance(suggestions1, list)
        assert isinstance(suggestions2, list)

    def test_generate_suggestions_with_different_companies(self):
        """Test génération de suggestions avec différentes entreprises"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des objets mock avec attributs
        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 45.0
        mock_booking.pickup_lon = 2.0
        mock_booking.dropoff_lat = 45.1
        mock_booking.dropoff_lon = 2.1
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 45.0
        mock_driver.current_lon = 2.0
        mock_driver.driver_type = Mock()
        mock_driver.driver_type.value = "REGULAR"
        mock_driver.user = None

        mock_assignment = Mock()
        mock_assignment.id = 1
        mock_assignment.booking = mock_booking
        mock_assignment.driver = mock_driver

        assignments = [mock_assignment]
        drivers = [mock_driver]

        # Test avec différentes entreprises
        suggestions1 = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-01-01",
        )

        suggestions2 = generator.generate_suggestions(
            company_id=2,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-01-01",
        )

        assert isinstance(suggestions1, list)
        assert isinstance(suggestions2, list)

    def test_generate_suggestions_with_empty_strings(self):
        """Test génération de suggestions avec chaînes vides"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des objets mock avec attributs
        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 45.0
        mock_booking.pickup_lon = 2.0
        mock_booking.dropoff_lat = 45.1
        mock_booking.dropoff_lon = 2.1
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 45.0
        mock_driver.current_lon = 2.0
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
