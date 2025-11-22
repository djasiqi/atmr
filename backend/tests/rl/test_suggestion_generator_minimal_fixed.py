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
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            assert generator.agent == mock_agent
            assert generator.model_path == "data/ml/dqn_agent_best_v33.pth"

    def test_lazy_import_rl_success(self):
        """Test import paresseux RL réussi"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Vérifier que l'agent est créé
            assert generator.agent == mock_agent

    def test_lazy_import_rl_failure(self):
        """Test import paresseux RL échoué"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent", side_effect=ImportError("RL not available")):
            generator = RLSuggestionGenerator()

            # L'agent devrait être None en cas d'erreur d'import
            assert generator.agent is None

    def test_load_model_file_not_found(self):
        """Test chargement de modèle - fichier non trouvé"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Mock os.path.exists pour retourner False
            with patch("os.path.exists", return_value=False):
                generator._load_model()

                # L'agent devrait rester inchangé
                assert generator.agent == mock_agent

    def test_load_model_with_exception(self):
        """Test chargement de modèle avec exception"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Mock os.path.exists pour retourner True
            with (
                patch("os.path.exists", return_value=True),
                patch("torch.load", side_effect=Exception("Load error")),
            ):
                generator._load_model()

                # L'agent devrait rester inchangé
                assert generator.agent == mock_agent

    def test_generate_suggestions_no_model(self):
        """Test génération de suggestions sans modèle"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()
            generator.agent = None  # Pas de modèle chargé

            # Créer des données de test
            assignments = [
                MockAssignment(1, MockBooking(1, 48.8566, 2.3522, 48.8606, 2.3372), MockDriver(1, 48.8566, 2.3522))
            ]
            drivers = [MockDriver(1, 48.8566, 2.3522)]

            suggestions = generator.generate_suggestions(
                company_id=1, assignments=assignments, drivers=drivers, for_date=datetime.now()
            )

            # Devrait retourner une liste vide
            assert suggestions == []

    def test_generate_suggestions_with_exception(self):
        """Test génération de suggestions avec exception"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Mock _generate_basic_suggestions pour lever une exception
            with patch.object(generator, "_generate_basic_suggestions", side_effect=Exception("Generation error")):
                suggestions = generator.generate_suggestions(
                    company_id=1, assignments=[], drivers=[], for_date=datetime.now()
                )

                # Devrait retourner une liste vide en cas d'erreur
                assert suggestions == []

    def test_generate_suggestions_with_parameters(self):
        """Test génération de suggestions avec paramètres"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Mock _generate_basic_suggestions pour retourner des suggestions
            mock_suggestions = [{"driver_id": 1, "booking_id": 1, "confidence": 0.8}]
            with patch.object(generator, "_generate_basic_suggestions", return_value=mock_suggestions):
                suggestions = generator.generate_suggestions(
                    company_id=1,
                    assignments=[],
                    drivers=[],
                    for_date=datetime.now(),
                    min_confidence=0.7,
                    max_suggestions=5,
                )

                assert suggestions == mock_suggestions

    def test_generate_suggestions_no_available_drivers(self):
        """Test génération de suggestions sans chauffeurs disponibles"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Créer des données de test sans chauffeurs disponibles
            assignments = [
                MockAssignment(1, MockBooking(1, 48.8566, 2.3522, 48.8606, 2.3372), MockDriver(1, 48.8566, 2.3522))
            ]
            drivers = [MockDriver(1, 48.8566, 2.3522)]
            drivers[0].is_available = False  # Chauffeur non disponible

            suggestions = generator.generate_suggestions(
                company_id=1, assignments=assignments, drivers=drivers, for_date=datetime.now()
            )

            # Devrait retourner une liste vide
            assert suggestions == []

    def test_generate_suggestions_no_unassigned_assignments(self):
        """Test génération de suggestions sans assignments non assignés"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Créer des données de test avec assignments déjà assignés
            assignments = []  # Pas d'assignments non assignés

            suggestions = generator.generate_suggestions(
                company_id=1, assignments=assignments, drivers=[], for_date=datetime.now()
            )

            # Devrait retourner une liste vide
            assert suggestions == []

    def test_generate_suggestions_with_confidence_threshold(self):
        """Test génération de suggestions avec seuil de confiance"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Mock _generate_basic_suggestions pour retourner des suggestions
            mock_suggestions = [
                {"driver_id": 1, "booking_id": 1, "confidence": 0.8},
                {"driver_id": 2, "booking_id": 2, "confidence": 0.6},
            ]
            with patch.object(generator, "_generate_basic_suggestions", return_value=mock_suggestions):
                suggestions = generator.generate_suggestions(
                    company_id=1, assignments=[], drivers=[], for_date=datetime.now(), min_confidence=0.7
                )

                # Devrait filtrer les suggestions avec confiance < 0.7
                assert len(suggestions) == 1
                assert suggestions[0]["confidence"] >= 0.7

    def test_generate_suggestions_max_suggestions(self):
        """Test génération de suggestions avec limite maximale"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Mock _generate_basic_suggestions pour retourner beaucoup de suggestions
            mock_suggestions = [{"driver_id": i, "booking_id": i, "confidence": 0.8} for i in range(20)]
            with patch.object(generator, "_generate_basic_suggestions", return_value=mock_suggestions):
                suggestions = generator.generate_suggestions(
                    company_id=1, assignments=[], drivers=[], for_date=datetime.now(), max_suggestions=5
                )

                # Devrait limiter à 5 suggestions
                assert len(suggestions) == 5

    def test_generate_suggestions_with_different_dates(self):
        """Test génération de suggestions avec différentes dates"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Test avec date passée
            past_date = datetime.now() - timedelta(days=1)
            suggestions = generator.generate_suggestions(company_id=1, assignments=[], drivers=[], for_date=past_date)

            # Devrait retourner une liste vide pour les dates passées
            assert suggestions == []

    def test_generate_suggestions_with_different_companies(self):
        """Test génération de suggestions avec différentes entreprises"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Test avec company_id négatif
            suggestions = generator.generate_suggestions(
                company_id=-1, assignments=[], drivers=[], for_date=datetime.now()
            )

            # Devrait retourner une liste vide pour company_id invalide
            assert suggestions == []

    def test_generate_suggestions_with_empty_strings(self):
        """Test génération de suggestions avec chaînes vides"""
        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            generator = RLSuggestionGenerator()

            # Test avec paramètres vides
            suggestions = generator.generate_suggestions(
                company_id=1, assignments=[], drivers=[], for_date=datetime.now(), min_confidence=0, max_suggestions=0
            )

            # Devrait retourner une liste vide
            assert suggestions == []
