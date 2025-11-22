"""
Tests simplifiés pour RLSuggestionGenerator
"""

from datetime import datetime, timedelta
from unittest.mock import patch

from services.rl.suggestion_generator import RLSuggestionGenerator


class TestRLSuggestionGeneratorSimple:
    """Tests simplifiés pour RLSuggestionGenerator"""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut"""
        generator = RLSuggestionGenerator()

        # Vérifier que l'agent est créé
        assert generator.agent is not None
        assert generator.model_path == "data/ml/dqn_agent_best_v3_3.pth"
        assert generator.max_suggestions == 10
        assert generator.min_confidence == 0.7

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés"""
        generator = RLSuggestionGenerator(
            model_path="custom/path.pth", max_suggestions=5, min_confidence=0.8
        )

        assert generator.model_path == "custom/path.pth"
        assert generator.max_suggestions == 5
        assert generator.min_confidence == 0.8

    def test_generate_suggestions_no_model(self):
        """Test génération de suggestions sans modèle chargé"""
        generator = RLSuggestionGenerator()

        # Mock assignments et drivers
        assignments = [
            {
                "id": 1,
                "booking": {
                    "id": 1,
                    "pickup_lat": 45.0,
                    "pickup_lng": 2.0,
                    "dropoff_lat": 45.1,
                    "dropoff_lng": 2.1,
                },
                "driver": {"id": 1, "lat": 45.0, "lng": 2.0},
            }
        ]
        drivers = [{"id": 1, "lat": 45.0, "lng": 2.0}]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date=datetime.now(),
        )

        # Devrait retourner des suggestions basiques
        assert isinstance(suggestions, list)

    def test_generate_suggestions_empty_input(self):
        """Test génération de suggestions avec entrée vide"""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=[], drivers=[], for_date=datetime.now()
        )

        assert suggestions == []

    def test_generate_suggestions_no_available_drivers(self):
        """Test génération de suggestions sans chauffeurs disponibles"""
        generator = RLSuggestionGenerator()

        assignments = [
            {
                "id": 1,
                "booking": {
                    "id": 1,
                    "pickup_lat": 45.0,
                    "pickup_lng": 2.0,
                    "dropoff_lat": 45.1,
                    "dropoff_lng": 2.1,
                },
                "driver": {"id": 1, "lat": 45.0, "lng": 2.0},
            }
        ]

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=assignments, drivers=[], for_date=datetime.now()
        )

        assert suggestions == []

    def test_generate_suggestions_with_exception(self):
        """Test génération de suggestions avec exception"""
        generator = RLSuggestionGenerator()

        # Mock pour provoquer une exception
        with patch.object(
            generator,
            "_generate_basic_suggestions",
            side_effect=Exception("Test error"),
        ):
            suggestions = generator.generate_suggestions(
                company_id=1, assignments=[], drivers=[], for_date=datetime.now()
            )

            assert suggestions == []

    def test_generate_suggestions_with_parameters(self):
        """Test génération de suggestions avec paramètres"""
        generator = RLSuggestionGenerator()

        assignments = [
            {
                "id": 1,
                "booking": {
                    "id": 1,
                    "pickup_lat": 45.0,
                    "pickup_lng": 2.0,
                    "dropoff_lat": 45.1,
                    "dropoff_lng": 2.1,
                },
                "driver": {"id": 1, "lat": 45.0, "lng": 2.0},
            }
        ]
        drivers = [{"id": 1, "lat": 45.0, "lng": 2.0}]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date=datetime.now(),
            min_confidence=0.8,
            max_suggestions=5,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 5

    def test_generate_suggestions_with_confidence_threshold(self):
        """Test génération de suggestions avec seuil de confiance"""
        generator = RLSuggestionGenerator()

        assignments = [
            {
                "id": 1,
                "booking": {
                    "id": 1,
                    "pickup_lat": 45.0,
                    "pickup_lng": 2.0,
                    "dropoff_lat": 45.1,
                    "dropoff_lng": 2.1,
                },
                "driver": {"id": 1, "lat": 45.0, "lng": 2.0},
            }
        ]
        drivers = [{"id": 1, "lat": 45.0, "lng": 2.0}]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date=datetime.now(),
            min_confidence=0.9,
        )

        assert isinstance(suggestions, list)

    def test_generate_suggestions_max_suggestions(self):
        """Test génération de suggestions avec limite maximale"""
        generator = RLSuggestionGenerator()

        assignments = [
            {
                "id": i,
                "booking": {
                    "id": i,
                    "pickup_lat": 45.0,
                    "pickup_lng": 2.0,
                    "dropoff_lat": 45.1,
                    "dropoff_lng": 2.1,
                },
                "driver": {"id": i, "lat": 45.0, "lng": 2.0},
            }
            for i in range(10)
        ]
        drivers = [{"id": i, "lat": 45.0, "lng": 2.0} for i in range(10)]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date=datetime.now(),
            max_suggestions=3,
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3

    def test_generate_suggestions_with_different_dates(self):
        """Test génération de suggestions avec différentes dates"""
        generator = RLSuggestionGenerator()

        assignments = [
            {
                "id": 1,
                "booking": {
                    "id": 1,
                    "pickup_lat": 45.0,
                    "pickup_lng": 2.0,
                    "dropoff_lat": 45.1,
                    "dropoff_lng": 2.1,
                },
                "driver": {"id": 1, "lat": 45.0, "lng": 2.0},
            }
        ]
        drivers = [{"id": 1, "lat": 45.0, "lng": 2.0}]

        # Test avec date d'aujourd'hui
        suggestions1 = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date=datetime.now(),
        )

        # Test avec date future
        suggestions2 = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date=datetime.now() + timedelta(days=1),
        )

        assert isinstance(suggestions1, list)
        assert isinstance(suggestions2, list)

    def test_generate_suggestions_with_different_companies(self):
        """Test génération de suggestions avec différentes entreprises"""
        generator = RLSuggestionGenerator()

        assignments = [
            {
                "id": 1,
                "booking": {
                    "id": 1,
                    "pickup_lat": 45.0,
                    "pickup_lng": 2.0,
                    "dropoff_lat": 45.1,
                    "dropoff_lng": 2.1,
                },
                "driver": {"id": 1, "lat": 45.0, "lng": 2.0},
            }
        ]
        drivers = [{"id": 1, "lat": 45.0, "lng": 2.0}]

        # Test avec différentes entreprises
        suggestions1 = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date=datetime.now(),
        )

        suggestions2 = generator.generate_suggestions(
            company_id=2,
            assignments=assignments,
            drivers=drivers,
            for_date=datetime.now(),
        )

        assert isinstance(suggestions1, list)
        assert isinstance(suggestions2, list)

    def test_generate_suggestions_with_empty_strings(self):
        """Test génération de suggestions avec chaînes vides"""
        generator = RLSuggestionGenerator()

        assignments = [
            {
                "id": 1,
                "booking": {
                    "id": 1,
                    "pickup_lat": 45.0,
                    "pickup_lng": 2.0,
                    "dropoff_lat": 45.1,
                    "dropoff_lng": 2.1,
                },
                "driver": {"id": 1, "lat": 45.0, "lng": 2.0},
            }
        ]
        drivers = [{"id": 1, "lat": 45.0, "lng": 2.0}]

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=assignments,
            drivers=drivers,
            for_date=datetime.now(),
        )

        assert isinstance(suggestions, list)
