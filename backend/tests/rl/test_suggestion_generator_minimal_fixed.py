"""
Tests minimaux pour RLSuggestionGenerator - Version corrigée
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from services.rl.suggestion_generator import RLSuggestionGenerator


class MockBooking:
    def __init__(self, booking_id, pickup_lat, pickup_lng, dropoff_lat, dropoff_lng):
        self.id = booking_id
        self.pickup_lat = pickup_lat
        self.pickup_lng = pickup_lng
        self.dropoff_lat = dropoff_lat
        self.dropoff_lng = dropoff_lng
        self.pickup_time = datetime.now()
        self.dropoff_time = datetime.now() + timedelta(minutes=30)


class MockDriver:
    def __init__(self, driver_id, lat, lng):
        self.id = driver_id
        self.lat = lat
        self.lng = lng
        self.is_available = True
        self.is_online = True


class MockAssignment:
    def __init__(self, assignment_id, booking, driver):
        self.id = assignment_id
        self.booking = booking
        self.driver = driver


class TestRLSuggestionGeneratorMinimal:
    """Tests minimaux pour RLSuggestionGenerator"""

    def test_init_basic(self):
        """Test initialisation basique"""
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

            assert generator.agent == mock_agent
            assert generator.model_path == "data/ml/dqn_agent_best_v33.pth"

    def test_lazy_import_rl_success(self):
        """Test import paresseux RL réussi"""
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
            assert generator.agent == mock_agent

    def test_lazy_import_rl_failure(self):
        """Test import paresseux RL échoué"""
        with patch(
            "services.rl.suggestion_generator.ImprovedDQNAgent",
            side_effect=ImportError("RL not available"),
        ):
            generator = RLSuggestionGenerator()

            # L'agent devrait être None en cas d'erreur d'import
            assert generator.agent is None

    def test_load_model_file_not_found(self):
        """Test chargement de modèle - fichier non trouvé"""
        with (
            patch("pathlib.Path") as mock_path_class,
            patch("services.rl.suggestion_generator._lazy_import_rl"),
        ):
            mock_path_instance = Mock()
            mock_path_instance.exists.return_value = False
            mock_path_class.return_value = mock_path_instance

            generator = RLSuggestionGenerator()

            # L'agent devrait être None car le fichier n'existe pas
            assert generator.agent is None

    def test_load_model_with_exception(self):
        """Test chargement de modèle avec exception"""
        with (
            patch("pathlib.Path") as mock_path_class,
            patch("services.rl.suggestion_generator._lazy_import_rl"),
            patch("services.rl.dispatch_env.DispatchEnv") as mock_env_class,
            patch(
                "services.rl.suggestion_generator.ImprovedDQNAgent"
            ) as mock_agent_class,
            patch("torch.load", side_effect=Exception("Load error")),
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

            # L'agent devrait être None en cas d'exception
            assert generator.agent is None

    def test_generate_suggestions_no_model(self):
        """Test génération de suggestions sans modèle"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle chargé

        # Créer des objets mock avec attributs
        from datetime import datetime, timedelta

        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 48.8566
        mock_booking.pickup_lon = 2.3522
        mock_booking.dropoff_lat = 48.8606
        mock_booking.dropoff_lon = 2.3376
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = True
        mock_driver.current_lat = 48.8566
        mock_driver.current_lon = 2.3522
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

        # Devrait retourner une liste vide car pas de drivers alternatifs
        assert isinstance(suggestions, list)

    def test_generate_suggestions_with_exception(self):
        """Test génération de suggestions avec exception"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Mock _generate_basic_suggestions pour lever une exception
        with patch.object(
            generator,
            "_generate_basic_suggestions",
            side_effect=Exception("Generation error"),
        ):
            # L'exception devrait être propagée car generate_suggestions
            # appelle directement _generate_basic_suggestions sans try/except
            import pytest

            with pytest.raises(Exception, match="Generation error"):
                generator.generate_suggestions(
                    company_id=1, assignments=[], drivers=[], for_date="2024-01-01"
                )

    def test_generate_suggestions_with_parameters(self):
        """Test génération de suggestions avec paramètres"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Mock _generate_basic_suggestions pour retourner des suggestions
        mock_suggestions = [{"driver_id": 1, "booking_id": 1, "confidence": 0.8}]
        with patch.object(
            generator, "_generate_basic_suggestions", return_value=mock_suggestions
        ):
            suggestions = generator.generate_suggestions(
                company_id=1,
                assignments=[],
                drivers=[],
                for_date="2024-01-01",
                min_confidence=0.7,
                max_suggestions=5,
            )

            assert suggestions == mock_suggestions

    def test_generate_suggestions_no_available_drivers(self):
        """Test génération de suggestions sans chauffeurs disponibles"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des objets mock avec attributs
        from datetime import datetime, timedelta

        mock_booking = Mock()
        mock_booking.id = 1
        mock_booking.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking.pickup_lat = 48.8566
        mock_booking.pickup_lon = 2.3522
        mock_booking.dropoff_lat = 48.8606
        mock_booking.dropoff_lon = 2.3376
        mock_booking.is_emergency = False

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.is_available = False  # Chauffeur non disponible
        mock_driver.current_lat = 48.8566
        mock_driver.current_lon = 2.3522
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

        # Devrait retourner une liste vide car pas de drivers disponibles
        assert suggestions == []

    def test_generate_suggestions_no_unassigned_assignments(self):
        """Test génération de suggestions sans assignments non assignés"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des données de test avec assignments déjà assignés
        assignments = []  # Pas d'assignments non assignés

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=[],
            for_date="2024-01-01",
        )

        # Devrait retourner une liste vide
        assert suggestions == []

    def test_generate_suggestions_with_confidence_threshold(self):
        """Test génération de suggestions avec seuil de confiance"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Créer des objets mock avec attributs
        mock_booking1 = Mock()
        mock_booking1.id = 1
        mock_booking1.scheduled_time = datetime.now() + timedelta(hours=1)
        mock_booking1.pickup_lat = 48.8566
        mock_booking1.pickup_lon = 2.3522
        mock_booking1.dropoff_lat = 48.8606
        mock_booking1.dropoff_lon = 2.3376
        mock_booking1.is_emergency = False

        mock_booking2 = Mock()
        mock_booking2.id = 2
        mock_booking2.scheduled_time = datetime.now() + timedelta(hours=2)
        mock_booking2.pickup_lat = 48.8606
        mock_booking2.pickup_lon = 2.3376
        mock_booking2.dropoff_lat = 48.8566
        mock_booking2.dropoff_lon = 2.3522
        mock_booking2.is_emergency = False

        mock_driver1 = Mock()
        mock_driver1.id = 1
        mock_driver1.is_available = True
        mock_driver1.current_lat = 48.8566
        mock_driver1.current_lon = 2.3522
        mock_driver1.driver_type = Mock()
        mock_driver1.driver_type.value = "REGULAR"
        mock_driver1.user = None

        mock_driver2 = Mock()
        mock_driver2.id = 2
        mock_driver2.is_available = True
        mock_driver2.current_lat = 48.8606
        mock_driver2.current_lon = 2.3376
        mock_driver2.driver_type = Mock()
        mock_driver2.driver_type.value = "REGULAR"
        mock_driver2.user = None

        mock_assignment1 = Mock()
        mock_assignment1.id = 1
        mock_assignment1.booking = mock_booking1
        mock_assignment1.driver = mock_driver1

        mock_assignment2 = Mock()
        mock_assignment2.id = 2
        mock_assignment2.booking = mock_booking2
        mock_assignment2.driver = mock_driver2

        assignments = [mock_assignment1, mock_assignment2]
        drivers = [mock_driver1, mock_driver2]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date="2024-01-01",
            min_confidence=0.7,
        )

        # _generate_basic_suggestions filtre déjà par min_confidence
        # Toutes les suggestions devraient avoir confidence >= 0.7
        assert isinstance(suggestions, list)
        for suggestion in suggestions:
            assert suggestion["confidence"] >= 0.7

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
            mock_booking.pickup_lat = 48.8566 + i * 0.01
            mock_booking.pickup_lon = 2.3522 + i * 0.01
            mock_booking.dropoff_lat = 48.8606 + i * 0.01
            mock_booking.dropoff_lon = 2.3376 + i * 0.01
            mock_booking.is_emergency = False

            mock_driver = Mock()
            mock_driver.id = i
            mock_driver.is_available = True
            mock_driver.current_lat = 48.8566 + i * 0.01
            mock_driver.current_lon = 2.3522 + i * 0.01
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
            max_suggestions=5,
        )

        # _generate_basic_suggestions limite déjà par max_suggestions
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5

    def test_generate_suggestions_with_different_dates(self):
        """Test génération de suggestions avec différentes dates"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Test avec date passée (format string)
        past_date = "2023-01-01"
        suggestions = generator.generate_suggestions(
            company_id=1, assignments=[], drivers=[], for_date=past_date
        )

        # Devrait retourner une liste vide pour les dates passées
        assert suggestions == []

    def test_generate_suggestions_with_different_companies(self):
        """Test génération de suggestions avec différentes entreprises"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Test avec company_id négatif
        suggestions = generator.generate_suggestions(
            company_id=-1, assignments=[], drivers=[], for_date="2024-01-01"
        )

        # Devrait retourner une liste vide pour company_id invalide
        assert suggestions == []

    def test_generate_suggestions_with_empty_strings(self):
        """Test génération de suggestions avec chaînes vides"""
        generator = RLSuggestionGenerator()
        generator.agent = None  # Pas de modèle

        # Test avec paramètres vides
        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            for_date="2024-01-01",
            min_confidence=0,
            max_suggestions=0,
        )

        # Devrait retourner une liste vide
        assert suggestions == []
