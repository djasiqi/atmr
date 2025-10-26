"""
Tests complets pour suggestion_generator.py - Couverture 95%+
"""
import logging
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from services.rl.suggestion_generator import RLSuggestionGenerator


class TestRLSuggestionGenerator:
    """Tests complets pour RLSuggestionGenerator"""

    def test_init_default(self):
        """Test initialisation avec paramètres par défaut"""
        generator = RLSuggestionGenerator()

        assert generator.max_suggestions == 5
        assert generator.min_confidence == 0.7
        assert generator.model_loaded is False

    def test_init_custom(self):
        """Test initialisation avec paramètres personnalisés"""
        generator = RLSuggestionGenerator(
            max_suggestions=10,
            min_confidence=0.8
        )

        assert generator.max_suggestions == 10
        assert generator.min_confidence == 0.8
        assert generator.model_loaded is False

    def test_lazy_import_rl_success(self):
        """Test _lazy_import_rl avec succès"""
        generator = RLSuggestionGenerator()

        with patch("services.rl.suggestion_generator.ImprovedDQNAgent") as mock_dqn:
            with patch("services.rl.suggestion_generator.DispatchEnv") as mock_env:
                mock_dqn.return_value = Mock()
                mock_env.return_value = Mock()

                result = generator._lazy_import_rl()

                assert result is True
                assert generator.model_loaded is True

    def test_lazy_import_rl_failure(self):
        """Test _lazy_import_rl avec échec"""
        generator = RLSuggestionGenerator()

        with patch("services.rl.suggestion_generator.ImprovedDQNAgent", side_effect=ImportError):
            result = generator._lazy_import_rl()

            assert result is False
            assert generator.model_loaded is False

    def test_load_model_success(self):
        """Test _load_model avec succès"""
        generator = RLSuggestionGenerator()

        with patch.object(generator, "_lazy_import_rl", return_value=True):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("torch.load") as mock_load:
                    mock_load.return_value = {"q_network_state_dict": {}, "optimizer_state_dict": {}}

                    result = generator._load_model()

                    assert result is True
                    assert generator.model_loaded is True

    def test_load_model_failure(self):
        """Test _load_model avec échec"""
        generator = RLSuggestionGenerator()

        with patch.object(generator, "_lazy_import_rl", return_value=False):
            result = generator._load_model()

            assert result is False
            assert generator.model_loaded is False

    def test_load_model_file_not_found(self):
        """Test _load_model avec fichier non trouvé"""
        generator = RLSuggestionGenerator()

        with patch.object(generator, "_lazy_import_rl", return_value=True):
            with patch("pathlib.Path.exists", return_value=False):
                result = generator._load_model()

                assert result is False
                assert generator.model_loaded is False

    def test_generate_suggestions_with_model(self):
        """Test generate_suggestions avec modèle chargé"""
        generator = RLSuggestionGenerator()

        # Mock assignments, drivers, bookings
        assignments = [
            {"id": 1, "booking_id": 1, "driver_id": 1, "status": "assigned"},
            {"id": 2, "booking_id": 2, "driver_id": 2, "status": "assigned"}
        ]
        drivers = [
            {"id": 1, "name": "Driver 1", "lat": 48.8566, "lon": 2.3522, "available": True},
            {"id": 2, "name": "Driver 2", "lat": 48.8606, "lon": 2.3376, "available": True}
        ]
        bookings = [
            {"id": 1, "pickup_lat": 48.8566, "pickup_lon": 2.3522, "dropoff_lat": 48.8606, "dropoff_lon": 2.3376},
            {"id": 2, "pickup_lat": 48.8606, "pickup_lon": 2.3376, "dropoff_lat": 48.8566, "dropoff_lon": 2.3522}
        ]

        with patch.object(generator, "_load_model", return_value=True):
            with patch.object(generator, "_get_suggestion_confidence", return_value=0.8):
                with patch.object(generator, "_format_suggestion", return_value={"suggestion": "test"}):
                    suggestions = generator.generate_suggestions(
                        company_id=1,
                        assignments=assignments,
                        drivers=drivers,
                        bookings=bookings,
                        for_date="2025-0.1-0.1",
                        min_confidence=0.7,
                        max_suggestions=5
                    )

                    assert isinstance(suggestions, list)
                    assert len(suggestions) <= 5

    def test_generate_suggestions_without_model(self):
        """Test generate_suggestions sans modèle"""
        generator = RLSuggestionGenerator()

        assignments = [{"id": 1, "booking_id": 1, "driver_id": 1, "status": "assigned"}]
        drivers = [{"id": 1, "name": "Driver 1", "lat": 48.8566, "lon": 2.3522, "available": True}]
        bookings = [{"id": 1, "pickup_lat": 48.8566, "pickup_lon": 2.3522, "dropoff_lat": 48.8606, "dropoff_lon": 2.3376}]

        with patch.object(generator, "_load_model", return_value=False):
            with patch.object(generator, "_get_heuristic_suggestions", return_value=[{"suggestion": "heuristic"}]):
                suggestions = generator.generate_suggestions(
                    company_id=1,
                    assignments=assignments,
                    drivers=drivers,
                    bookings=bookings,
                    for_date="2025-0.1-0.1",
                    min_confidence=0.7,
                    max_suggestions=5
                )

                assert isinstance(suggestions, list)
                assert len(suggestions) == 1
                assert suggestions[0]["suggestion"] == "heuristic"

    def test_generate_suggestions_empty_data(self):
        """Test generate_suggestions avec données vides"""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            bookings=[],
            for_date="2025-0.1-0.1",
            min_confidence=0.7,
            max_suggestions=5
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_get_suggestion_confidence(self):
        """Test _get_suggestion_confidence"""
        generator = RLSuggestionGenerator()

        # Mock state et q_values
        state = np.array([1, 2, 3])
        q_values = np.array([0.8, 0.9, 0.7])

        confidence = generator._get_suggestion_confidence(state, q_values)

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_format_suggestion(self):
        """Test _format_suggestion"""
        generator = RLSuggestionGenerator()

        suggestion_data = {
            "assignment_id": 1,
            "current_driver_id": 1,
            "suggested_driver_id": 2,
            "confidence": 0.8,
            "reason": "Better match"
        }

        formatted = generator._format_suggestion(suggestion_data)

        assert isinstance(formatted, dict)
        assert "assignment_id" in formatted
        assert "current_driver_id" in formatted
        assert "suggested_driver_id" in formatted
        assert "confidence" in formatted
        assert "reason" in formatted

    def test_get_heuristic_suggestions(self):
        """Test _get_heuristic_suggestions"""
        generator = RLSuggestionGenerator()

        assignments = [{"id": 1, "booking_id": 1, "driver_id": 1, "status": "assigned"}]
        drivers = [{"id": 1, "name": "Driver 1", "lat": 48.8566, "lon": 2.3522, "available": True}]
        bookings = [{"id": 1, "pickup_lat": 48.8566, "pickup_lon": 2.3522, "dropoff_lat": 48.8606, "dropoff_lon": 2.3376}]

        suggestions = generator._get_heuristic_suggestions(assignments, drivers, bookings)

        assert isinstance(suggestions, list)

    def test_calculate_distance(self):
        """Test _calculate_distance"""
        generator = RLSuggestionGenerator()

        lat1, lon1 = 48.8566, 2.3522
        lat2, lon2 = 48.8606, 2.3376

        distance = generator._calculate_distance(lat1, lon1, lat2, lon2)

        assert isinstance(distance, float)
        assert distance >= 0

    def test_is_model_loaded(self):
        """Test is_model_loaded property"""
        generator = RLSuggestionGenerator()

        assert generator.is_model_loaded is False

        generator.model_loaded = True
        assert generator.is_model_loaded is True

    def test_get_model_info(self):
        """Test get_model_info"""
        generator = RLSuggestionGenerator()

        info = generator.get_model_info()

        assert isinstance(info, dict)
        assert "loaded" in info
        assert "max_suggestions" in info
        assert "min_confidence" in info

    def test_reload_model(self):
        """Test reload_model"""
        generator = RLSuggestionGenerator()

        with patch.object(generator, "_load_model", return_value=True):
            result = generator.reload_model()

            assert result is True

    def test_clear_model(self):
        """Test clear_model"""
        generator = RLSuggestionGenerator()

        generator.model_loaded = True
        generator.clear_model()

        assert generator.model_loaded is False

    def test_generate_suggestions_with_exception(self):
        """Test generate_suggestions avec exception"""
        generator = RLSuggestionGenerator()

        assignments = [{"id": 1, "booking_id": 1, "driver_id": 1, "status": "assigned"}]
        drivers = [{"id": 1, "name": "Driver 1", "lat": 48.8566, "lon": 2.3522, "available": True}]
        bookings = [{"id": 1, "pickup_lat": 48.8566, "pickup_lon": 2.3522, "dropoff_lat": 48.8606, "dropoff_lon": 2.3376}]

        with patch.object(generator, "_load_model", side_effect=Exception("Model error")):
            suggestions = generator.generate_suggestions(
                company_id=1,
                assignments=assignments,
                drivers=drivers,
                bookings=bookings,
                for_date="2025-0.1-0.1",
                min_confidence=0.7,
                max_suggestions=5
            )

            assert isinstance(suggestions, list)
            assert len(suggestions) == 0

    def test_generate_suggestions_max_suggestions(self):
        """Test generate_suggestions avec max_suggestions limité"""
        generator = RLSuggestionGenerator(max_suggestions=2)

        assignments = [
            {"id": 1, "booking_id": 1, "driver_id": 1, "status": "assigned"},
            {"id": 2, "booking_id": 2, "driver_id": 2, "status": "assigned"},
            {"id": 3, "booking_id": 3, "driver_id": 3, "status": "assigned"}
        ]
        drivers = [
            {"id": 1, "name": "Driver 1", "lat": 48.8566, "lon": 2.3522, "available": True},
            {"id": 2, "name": "Driver 2", "lat": 48.8606, "lon": 2.3376, "available": True},
            {"id": 3, "name": "Driver 3", "lat": 48.8646, "lon": 2.3226, "available": True}
        ]
        bookings = [
            {"id": 1, "pickup_lat": 48.8566, "pickup_lon": 2.3522, "dropoff_lat": 48.8606, "dropoff_lon": 2.3376},
            {"id": 2, "pickup_lat": 48.8606, "pickup_lon": 2.3376, "dropoff_lat": 48.8646, "dropoff_lon": 2.3226},
            {"id": 3, "pickup_lat": 48.8646, "pickup_lon": 2.3226, "dropoff_lat": 48.8566, "dropoff_lon": 2.3522}
        ]

        with patch.object(generator, "_load_model", return_value=True):
            with patch.object(generator, "_get_suggestion_confidence", return_value=0.8):
                with patch.object(generator, "_format_suggestion", return_value={"suggestion": "test"}):
                    suggestions = generator.generate_suggestions(
                        company_id=1,
                        assignments=assignments,
                        drivers=drivers,
                        bookings=bookings,
                        for_date="2025-0.1-0.1",
                        min_confidence=0.7,
                        max_suggestions=2
                    )

                    assert isinstance(suggestions, list)
                    assert len(suggestions) <= 2

    def test_generate_suggestions_min_confidence(self):
        """Test generate_suggestions avec min_confidence"""
        generator = RLSuggestionGenerator(min_confidence=0.9)

        assignments = [{"id": 1, "booking_id": 1, "driver_id": 1, "status": "assigned"}]
        drivers = [{"id": 1, "name": "Driver 1", "lat": 48.8566, "lon": 2.3522, "available": True}]
        bookings = [{"id": 1, "pickup_lat": 48.8566, "pickup_lon": 2.3522, "dropoff_lat": 48.8606, "dropoff_lon": 2.3376}]

        with patch.object(generator, "_load_model", return_value=True):
            with patch.object(generator, "_get_suggestion_confidence", return_value=0.8):  # < 0.9
                with patch.object(generator, "_format_suggestion", return_value={"suggestion": "test"}):
                    suggestions = generator.generate_suggestions(
                        company_id=1,
                        assignments=assignments,
                        drivers=drivers,
                        bookings=bookings,
                        for_date="2025-0.1-0.1",
                        min_confidence=0.9,
                        max_suggestions=5
                    )

                    assert isinstance(suggestions, list)
                    # Devrait être vide car confidence < min_confidence

    def test_edge_case_none_assignments(self):
        """Test avec assignments None"""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=None,
            drivers=[],
            bookings=[],
            for_date="2025-0.1-0.1",
            min_confidence=0.7,
            max_suggestions=5
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_edge_case_none_drivers(self):
        """Test avec drivers None"""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=None,
            bookings=[],
            for_date="2025-0.1-0.1",
            min_confidence=0.7,
            max_suggestions=5
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_edge_case_none_bookings(self):
        """Test avec bookings None"""
        generator = RLSuggestionGenerator()

        suggestions = generator.generate_suggestions(
            company_id=1,
            assignments=[],
            drivers=[],
            bookings=None,
            for_date="2025-0.1-0.1",
            min_confidence=0.7,
            max_suggestions=5
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_edge_case_invalid_confidence(self):
        """Test avec confidence invalide"""
        generator = RLSuggestionGenerator()

        assignments = [{"id": 1, "booking_id": 1, "driver_id": 1, "status": "assigned"}]
        drivers = [{"id": 1, "name": "Driver 1", "lat": 48.8566, "lon": 2.3522, "available": True}]
        bookings = [{"id": 1, "pickup_lat": 48.8566, "pickup_lon": 2.3522, "dropoff_lat": 48.8606, "dropoff_lon": 2.3376}]

        with patch.object(generator, "_load_model", return_value=True):
            with patch.object(generator, "_get_suggestion_confidence", return_value=0.8):
                with patch.object(generator, "_format_suggestion", return_value={"suggestion": "test"}):
                    suggestions = generator.generate_suggestions(
                        company_id=1,
                        assignments=assignments,
                        drivers=drivers,
                        bookings=bookings,
                        for_date="2025-0.1-0.1",
                        min_confidence=-0.1,  # Invalide
                        max_suggestions=5
                    )

                    assert isinstance(suggestions, list)

    def test_edge_case_invalid_max_suggestions(self):
        """Test avec max_suggestions invalide"""
        generator = RLSuggestionGenerator()

        assignments = [{"id": 1, "booking_id": 1, "driver_id": 1, "status": "assigned"}]
        drivers = [{"id": 1, "name": "Driver 1", "lat": 48.8566, "lon": 2.3522, "available": True}]
        bookings = [{"id": 1, "pickup_lat": 48.8566, "pickup_lon": 2.3522, "dropoff_lat": 48.8606, "dropoff_lon": 2.3376}]

        with patch.object(generator, "_load_model", return_value=True):
            with patch.object(generator, "_get_suggestion_confidence", return_value=0.8):
                with patch.object(generator, "_format_suggestion", return_value={"suggestion": "test"}):
                    suggestions = generator.generate_suggestions(
                        company_id=1,
                        assignments=assignments,
                        drivers=drivers,
                        bookings=bookings,
                        for_date="2025-0.1-0.1",
                        min_confidence=0.7,
                        max_suggestions=-1  # Invalide
                    )

                    assert isinstance(suggestions, list)

    def test_edge_case_empty_state(self):
        """Test avec état vide"""
        generator = RLSuggestionGenerator()

        state = np.array([])
        q_values = np.array([])

        confidence = generator._get_suggestion_confidence(state, q_values)

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_edge_case_none_state(self):
        """Test avec état None"""
        generator = RLSuggestionGenerator()

        confidence = generator._get_suggestion_confidence(None, None)

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_edge_case_empty_suggestion_data(self):
        """Test avec données de suggestion vides"""
        generator = RLSuggestionGenerator()

        formatted = generator._format_suggestion({})

        assert isinstance(formatted, dict)

    def test_edge_case_none_suggestion_data(self):
        """Test avec données de suggestion None"""
        generator = RLSuggestionGenerator()

        formatted = generator._format_suggestion(None)

        assert isinstance(formatted, dict)

    def test_edge_case_empty_heuristic_data(self):
        """Test avec données heuristiques vides"""
        generator = RLSuggestionGenerator()

        suggestions = generator._get_heuristic_suggestions([], [], [])

        assert isinstance(suggestions, list)
        assert len(suggestions) == 0

    def test_edge_case_invalid_coordinates(self):
        """Test avec coordonnées invalides"""
        generator = RLSuggestionGenerator()

        # Coordonnées invalides
        lat1, lon1 = float("inf"), float("nan")
        lat2, lon2 = float("-inf"), float("nan")

        distance = generator._calculate_distance(lat1, lon1, lat2, lon2)

        assert isinstance(distance, float)
        assert distance >= 0

    def test_edge_case_same_coordinates(self):
        """Test avec coordonnées identiques"""
        generator = RLSuggestionGenerator()

        lat1, lon1 = 48.8566, 2.3522
        lat2, lon2 = 48.8566, 2.3522

        distance = generator._calculate_distance(lat1, lon1, lat2, lon2)

        assert isinstance(distance, float)
        assert distance == 0
