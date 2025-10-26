#!/usr/bin/env python3

# Constantes pour éviter les valeurs magiques
from pathlib import Path
DRIVERS_ZERO = 0
REWARD_ZERO = 0
ACTION_ZERO = 0
INVALIDES_ONE = 1
TOTAL_ACTIONS_ZERO = 0
EST_ONE = 1
INVALID_PERCENTAGE_ONE = 1

"""
Tests pour les cas limites d'Action Masking.

Vérifie que le système gère correctement les cas extrêmes :
- 0 driver / 0 booking
- drivers > DRIVERS_ZERO / bookings = DRIVERS_ZERO
- fenêtres temporelles impossibles
- index out of range

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import conditionnel pour éviter les erreurs si les modules ne sont pas disponibles
try:
    from services.rl.dispatch_env import DispatchEnv
    from services.rl.improved_dqn_agent import ImprovedDQNAgent
except ImportError:
    DispatchEnv = None
    ImprovedDQNAgent = None

logger = logging.getLogger(__name__)


class TestActionMaskingEdges:
    """Tests pour les cas limites d'Action Masking."""

    @pytest.fixture
    def mock_env(self):
        """Créer un environnement mock pour les tests."""
        if DispatchEnv is None:
            pytest.skip("DispatchEnv non disponible")

        return DispatchEnv(num_drivers=3, max_bookings=5, simulation_hours=1)

    @pytest.fixture
    def mock_agent(self):
        """Créer un agent mock pour les tests."""
        if ImprovedDQNAgent is None:
            pytest.skip("ImprovedDQNAgent non disponible")

        return ImprovedDQNAgent(
            state_dim=0.100,
            action_dim=16,  # 3 drivers * 5 bookings + 1 wait
            learning_rate=0.0001,
            epsilon_start=0.1
        )

    def test_zero_drivers_zero_bookings(self, mock_env):
        """Test avec 0 drivers et 0 bookings."""
        # Initialiser l'environnement avec 0 drivers et 0 bookings
        mock_env.num_drivers = 0
        mock_env.max_bookings = 0
        mock_env.action_space.n = 1  # Seulement l'action wait

        # Réinitialiser les listes
        mock_env.drivers = []
        mock_env.bookings = []

        # Tester get_valid_actions
        valid_actions = mock_env.get_valid_actions()
        assert valid_actions == [0], f"Devrait retourner [0], a retourné {valid_actions}"

        # Tester le masque d'actions
        mask = mock_env._get_valid_actions_mask()
        assert mask.shape == (1,), f"Masque devrait avoir shape (1,), a {mask.shape}"
        assert mask[0], "Action 0 (wait) devrait être valide"

        # Tester step avec action 0
        _state, reward, _terminated, _truncated, info = mock_env.step(0)
        assert reward <= REWARD_ZERO, "Reward devrait être négatif ou nul pour inaction"
        assert not info.get("invalid_action", False), "Action 0 devrait être valide"

        logger.info("✅ Test 0 drivers/0 bookings réussi")

    def test_drivers_available_zero_bookings(self, mock_env):
        """Test avec drivers disponibles mais 0 bookings."""
        # Initialiser avec drivers mais pas de bookings
        mock_env.drivers = [
            {"id": 0, "lat": 46.2, "lon": 6.1, "available": True, "load": 0},
            {"id": 1, "lat": 46.2, "lon": 6.1, "available": True, "load": 0},
        ]
        mock_env.bookings = []

        # Tester get_valid_actions
        valid_actions = mock_env.get_valid_actions()
        assert 0 in valid_actions, "Action wait devrait être disponible"

        # Tester le masque d'actions
        mask = mock_env._get_valid_actions_mask()
        assert mask[0], "Action 0 (wait) devrait être valide"

        # Vérifier qu'aucune action d'assignation n'est valide
        assignment_actions = [i for i in range(1, mock_env.action_space.n) if mask[i]]
        assert len(assignment_actions) == 0, f"Pas d'actions d'assignation valides, trouvé {assignment_actions}"

        # Tester step avec action 0
        _state, _reward, _terminated, _truncated, info = mock_env.step(0)
        assert not info.get("invalid_action", False), "Action 0 devrait être valide"

        logger.info("✅ Test drivers disponibles/0 bookings réussi")

    def test_zero_drivers_bookings_available(self, mock_env):
        """Test avec 0 drivers mais bookings disponibles."""
        # Initialiser avec bookings mais pas de drivers
        mock_env.drivers = []
        mock_env.bookings = [
            {
                "id": 0,
                "pickup_lat": 46.2,
                "pickup_lon": 6.1,
                "priority": 3,
                "time_window_end": 100,
                "time_remaining": 30,
                "assigned": False
            }
        ]

        # Tester get_valid_actions
        valid_actions = mock_env.get_valid_actions()
        assert valid_actions == [0], f"Devrait retourner [0], a retourné {valid_actions}"

        # Tester le masque d'actions
        mask = mock_env._get_valid_actions_mask()
        assert mask[0], "Action 0 (wait) devrait être valide"

        # Vérifier qu'aucune action d'assignation n'est valide
        assignment_actions = [i for i in range(1, mock_env.action_space.n) if mask[i]]
        assert len(assignment_actions) == 0, f"Pas d'actions d'assignation valides, trouvé {assignment_actions}"

        logger.info("✅ Test 0 drivers/bookings disponibles réussi")

    def test_impossible_time_windows(self, mock_env):
        """Test avec fenêtres temporelles impossibles."""
        # Créer des bookings avec des fenêtres temporelles impossibles
        mock_env.drivers = [
            {"id": 0, "lat": 46.2, "lon": 6.1, "available": True, "load": 0}
        ]
        mock_env.bookings = [
            {
                "id": 0,
                "pickup_lat": 46.2,
                "pickup_lon": 6.1,
                "priority": 3,
                "time_window_end": 0,  # Fenêtre expirée
                "time_remaining": -10,
                "assigned": False
            },
            {
                "id": 1,
                "pickup_lat": 46.3,  # Très loin
                "pickup_lon": 6.2,
                "priority": 3,
                "time_window_end": 5,  # Fenêtre très courte
                "time_remaining": 5,
                "assigned": False
            }
        ]

        # Tester get_valid_actions
        valid_actions = mock_env.get_valid_actions()
        assert 0 in valid_actions, "Action wait devrait être disponible"

        # Vérifier qu'aucune action d'assignation n'est valide à cause des contraintes
        mask = mock_env._get_valid_actions_mask()
        assignment_actions = [i for i in range(1, mock_env.action_space.n) if mask[i]]
        assert len(assignment_actions) == 0, f"Pas d'actions d'assignation valides avec fenêtres impossibles, trouvé {assignment_actions}"

        logger.info("✅ Test fenêtres temporelles impossibles réussi")

    def test_index_out_of_range_protection(self, mock_env):
        """Test protection contre les index out of range."""
        # Créer une situation où l'index pourrait dépasser les limites
        mock_env.drivers = [{"id": 0, "lat": 46.2, "lon": 6.1, "available": True, "load": 0}]
        mock_env.bookings = [
            {
                "id": 0,
                "pickup_lat": 46.2,
                "pickup_lon": 6.1,
                "priority": 3,
                "time_window_end": 100,
                "time_remaining": 30,
                "assigned": False
            }
        ]

        # Tester avec des actions qui pourraient causer des index out of range
        invalid_actions = [mock_env.action_space.n, mock_env.action_space.n + 1, -1]

        for invalid_action in invalid_actions:
            _state, reward, _terminated, _truncated, info = mock_env.step(invalid_action)
            assert reward < REWARD_ZERO, f"Action invalide {invalid_action} devrait avoir reward négatif"
            assert info.get("invalid_action", False), f"Action {invalid_action} devrait être marquée invalide"

        logger.info("✅ Test protection index out of range réussi")

    def test_agent_fallback_with_empty_valid_actions(self, mock_agent):
        """Test que l'agent utilise le fallback quand valid_actions est vide."""
        if mock_agent is None:
            pytest.skip("ImprovedDQNAgent non disponible")

        # Créer un état mock
        state = np.random.random(mock_agent.state_dim)

        # Tester avec valid_actions vide
        with patch("logging.warning") as mock_warning:
            action = mock_agent.select_action(state, valid_actions=[])

            # Vérifier que l'action est 0 (wait)
            assert action == ACTION_ZERO, f"Devrait retourner action ACTION_ZERO, a retourné {action}"

            # Vérifier qu'un warning a été loggé
            mock_warning.assert_called()
            warning_call = mock_warning.call_args[0][0]
            assert "valid_actions vide" in warning_call, "Devrait logger un warning sur valid_actions vide"

        logger.info("✅ Test agent fallback avec valid_actions vide réussi")

    def test_agent_fallback_with_invalid_actions(self, mock_agent):
        """Test que l'agent gère les actions invalides."""
        if mock_agent is None:
            pytest.skip("ImprovedDQNAgent non disponible")

        # Créer un état mock
        state = np.random.random(mock_agent.state_dim)

        # Tester avec des actions invalides (hors limites)
        invalid_actions = [mock_agent.action_dim, mock_agent.action_dim + 1, -1]

        for invalid_action in invalid_actions:
            # L'agent devrait gérer cela gracieusement
            try:
                result_action = mock_agent.select_action(state, valid_actions=[invalid_action])
                # Si pas d'erreur, vérifier que l'action est dans les limites
                assert 0 <= result_action < mock_agent.action_dim, f"Action {result_action} hors limites"
            except Exception as e:
                pytest.fail(f"Agent a échoué avec action invalide {invalid_action}: {e}")

        logger.info("✅ Test agent avec actions invalides réussi")

    def test_edge_case_integration(self, ____________________________________________________________________________________________________mock_env, mock_agent):
        """Test d'intégration avec cas limites."""
        if mock_agent is None:
            pytest.skip("ImprovedDQNAgent non disponible")

        # Test avec différents scénarios limites
        test_scenarios = [
            {"drivers": [], "bookings": []},
            {"drivers": [{"id": 0, "lat": 46.2, "lon": 6.1, "available": True, "load": 0}], "bookings": []},
            {"drivers": [], "bookings": [{"id": 0, "pickup_lat": 46.2, "pickup_lon": 6.1, "priority": 3, "time_window_end": 100, "time_remaining": 30, "assigned": False}]},
        ]

        for i, scenario in enumerate(test_scenarios):
            mock_env.drivers = scenario["drivers"]
            mock_env.bookings = scenario["bookings"]

            # Obtenir l'état et les actions valides
            state = mock_env._get_observation()
            valid_actions = mock_env.get_valid_actions()

            # Tester que l'agent peut sélectionner une action
            try:
                action = mock_agent.select_action(state, valid_actions)
                assert 0 <= action < mock_env.action_space.n, f"Scénario {i}: Action {action} hors limites"

                # Tester que l'environnement peut traiter l'action
                _next_state, _reward, _terminated, _truncated, info = mock_env.step(action)
                assert not info.get("invalid_action", False), f"Scénario {i}: Action {action} marquée invalide"

            except Exception as e:
                pytest.fail(f"Scénario {i} a échoué: {e}")

        logger.info("✅ Test intégration cas limites réussi")


def test_action_masking_performance():
    """Test de performance pour vérifier que les actions invalides < INVALIDES_ONE%."""
    if DispatchEnv is None or ImprovedDQNAgent is None:
        pytest.skip("Modules non disponibles")

    # Créer environnement et agent
    env = DispatchEnv(num_drivers=5, max_bookings=10, simulation_hours=1)
    agent = ImprovedDQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=0.0001,
        epsilon_start=0.1
    )

    # Simuler plusieurs épisodes
    invalid_actions_count = 0
    total_actions = 0

    for _episode in range(10):
        state, _ = env.reset()

        for _step in range(50):  # 50 steps par épisode
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)

            next_state, _reward, terminated, truncated, info = env.step(action)

            total_actions += 1
            if info.get("invalid_action", False):
                invalid_actions_count += 1

            state = next_state

            if terminated or truncated:
                break

    # Calculer le pourcentage d'actions invalides
    invalid_percentage = (invalid_actions_count / total_actions) * 1 if total_actions > TOTAL_ACTIONS_ZERO else TOTAL_ACTIONS_ZERO

    # Vérifier que le pourcentage est < EST_ONE%
    assert invalid_percentage < INVALID_PERCENTAGE_ONE, f"Pourcentage d'actions invalides trop élevé: {invalid_percentage"

    logger.info("✅ Performance test réussi: %s% d'actions invalides", invalid_percentage)


if __name__ == "__main__":
    # Exécuter les tests si le script est appelé directement
    pytest.main([__file__, "-v"])
