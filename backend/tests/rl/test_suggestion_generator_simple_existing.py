"""Tests simples pour RLSuggestionGenerator - méthodes existantes seulement."""

from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from services.rl.suggestion_generator import RLSuggestionGenerator


class TestRLSuggestionGeneratorSimple:
    """Tests simples pour RLSuggestionGenerator."""

    def test_init_basic(self):
        """Test initialisation basique."""
        generator = RLSuggestionGenerator()

        assert generator.model_path is not None
        assert generator.agent is None  # Pas chargé à cause de l'erreur de modèle

    def test_init_with_custom_path(self):
        """Test initialisation avec chemin personnalisé."""
        generator = RLSuggestionGenerator(model_path="custom/path.pth")

        assert generator.model_path == "custom/path.pth"

    def test_generate_suggestions_basic(self):
        """Test génération suggestions basique."""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1, assignments=[], drivers=[], for_date="2024-0.1-0.1", min_confidence=0.5, max_suggestions=5
        )

        assert isinstance(suggestions, list)

    def test_generate_suggestions_with_data(self):
        """Test génération suggestions avec données."""
        generator = RLSuggestionGenerator()

        # Mock assignments et drivers
        mock_assignment = Mock()
        mock_assignment.booking = Mock()
        mock_assignment.driver = Mock()
        mock_assignment.booking.id = 1
        mock_assignment.driver.id = 1

        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.user = Mock()
        mock_driver.user.first_name = "John"
        mock_driver.user.last_name = "Doe"

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[mock_assignment],
            drivers=[mock_driver],
            for_date="2024-0.1-0.1",
            min_confidence=0.5,
            max_suggestions=5,
        )

        assert isinstance(suggestions, list)

    def test_generate_rl_suggestions_no_agent(self):
        """Test génération RL suggestions sans agent."""
        generator = RLSuggestionGenerator()
        generator.agent = None

        suggestions = generator._generate_rl_suggestions(
            company_id=1, assignments=[], drivers=[], for_date="2024-0.1-0.1", min_confidence=0.5, max_suggestions=5
        )

        assert suggestions == []

    def test_generate_rl_suggestions_with_agent(self):
        """Test génération RL suggestions avec agent."""
        generator = RLSuggestionGenerator()

        # Mock agent
        mock_agent = Mock()
        mock_q_values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
        mock_agent.q_network.return_value = mock_q_values
        generator.agent = mock_agent

        # Mock assignment
        mock_assignment = Mock()
        mock_assignment.booking = Mock()
        mock_assignment.driver = Mock()
        mock_assignment.booking.id = 1
        mock_assignment.driver.id = 1

        # Mock driver
        mock_driver = Mock()
        mock_driver.id = 1
        mock_driver.user = Mock()
        mock_driver.user.first_name = "John"
        mock_driver.user.last_name = "Doe"

        suggestions = generator._generate_rl_suggestions(
            company_id=1,
            assignments=[mock_assignment],
            drivers=[mock_driver],
            for_date="2024-0.1-0.1",
            min_confidence=0.5,
            max_suggestions=5,
        )

        assert isinstance(suggestions, list)

    def test_build_state_basic(self):
        """Test construction état basique."""
        generator = RLSuggestionGenerator()

        # Mock assignment
        mock_assignment = Mock()
        mock_booking = Mock()
        mock_booking.scheduled_time = datetime.now()
        mock_booking.pickup_lat = 48.8566
        mock_booking.pickup_lon = 2.3522
        mock_booking.dropoff_lat = 48.8606
        mock_booking.dropoff_lon = 2.3376
        mock_booking.is_emergency = False
        mock_assignment.booking = mock_booking

        # Mock drivers avec valeurs numériques réelles
        mock_drivers = []
        for i in range(5):
            mock_driver = Mock()
            mock_driver.id = i
            mock_driver.lat = 48.8566 + i * 0.0001
            mock_driver.lon = 2.3522 + i * 0.0001
            mock_driver.available = True
            mock_driver.assignments = []
            mock_drivers.append(mock_driver)

        # Mock haversine_distance pour éviter l'erreur
        with patch("shared.geo_utils.haversine_distance") as mock_haversine:
            mock_haversine.return_value = 1.0  # Distance fixe

            state = generator._build_state(mock_assignment, mock_drivers)

            assert isinstance(state, np.ndarray)
            assert len(state) == 19  # 4 booking features + 5*3 driver features

    def test_calculate_confidence(self):
        """Test calcul confiance."""
        generator = RLSuggestionGenerator()

        confidence = generator._calculate_confidence(0.5, 0)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_confidence_edge_cases(self):
        """Test calcul confiance cas limites."""
        generator = RLSuggestionGenerator()

        # Test avec Q-value négative
        confidence_neg = generator._calculate_confidence(-0.5, 0)
        assert isinstance(confidence_neg, float)

        # Test avec Q-value élevée
        confidence_high = generator._calculate_confidence(2.0, 0)
        assert isinstance(confidence_high, float)

        # Test avec rang élevé
        confidence_rank = generator._calculate_confidence(0.5, 5)
        assert isinstance(confidence_rank, float)

    def test_generate_basic_suggestions(self):
        """Test génération suggestions basiques."""
        generator = RLSuggestionGenerator()

        # Mock assignments
        mock_assignments = []
        for i in range(3):
            mock_assignment = Mock()
            mock_assignment.id = i
            mock_assignment.booking = Mock()
            mock_assignment.booking.id = i
            mock_assignment.driver = Mock()
            mock_assignment.driver.id = i
            mock_assignments.append(mock_assignment)

        # Mock drivers
        mock_drivers = []
        for i in range(3):
            mock_driver = Mock()
            mock_driver.id = i
            mock_driver.lat = 48.8566 + i * 0.0001
            mock_driver.lon = 2.3522 + i * 0.0001
            mock_driver.available = True
            mock_drivers.append(mock_driver)

        suggestions = generator._generate_basic_suggestions(mock_assignments, mock_drivers, 0.5, 5)

        assert isinstance(suggestions, list)

    def test_generate_basic_suggestions_empty(self):
        """Test génération suggestions basiques - données vides."""
        generator = RLSuggestionGenerator()

        suggestions = generator._generate_basic_suggestions([], [], 0.5, 5)

        assert suggestions == []

    def test_generate_rl_suggestions_exception(self):
        """Test génération RL suggestions avec exception."""
        generator = RLSuggestionGenerator()

        # Mock agent qui lève une exception
        mock_agent = Mock()
        mock_agent.q_network.side_effect = Exception("Test error")
        generator.agent = mock_agent

        # Mock assignment
        mock_assignment = Mock()
        mock_assignment.booking = Mock()
        mock_assignment.driver = Mock()
        mock_assignment.booking.id = 1
        mock_assignment.driver.id = 1

        suggestions = generator._generate_rl_suggestions(
            company_id=1,
            assignments=[mock_assignment],
            drivers=[],
            for_date="2024-0.1-0.1",
            min_confidence=0.5,
            max_suggestions=5,
        )

        # Devrait fallback vers suggestions basiques
        assert isinstance(suggestions, list)
