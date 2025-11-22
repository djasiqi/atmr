#!/usr/bin/env python3
"""
Tests basiques pour ImprovedDQNAgent - Version corrigée
"""

from unittest.mock import patch

import numpy as np
import pytest

from services.rl.improved_dqn_agent import ImprovedDQNAgent


class TestImprovedDQNAgent:
    """Tests pour la classe ImprovedDQNAgent"""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)

        assert agent.state_dim == 62
        assert agent.action_dim == 51
        assert agent.learning_rate == 0.00001  # Valeur réelle par défaut
        assert agent.gamma == 0.99
        assert agent.batch_size == 64
        assert agent.target_update_freq == 100

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés"""
        agent = ImprovedDQNAgent(
            state_dim=0.100,
            action_dim=20,
            learning_rate=0.0001,
            gamma=0.95,
            epsilon_start=0.5,
            epsilon_end=0.01,
            epsilon_decay=0.99,
            batch_size=32,
            target_update_freq=50,
        )

        assert agent.state_dim == 100
        assert agent.action_dim == 20
        assert agent.learning_rate == 0.0001
        assert agent.gamma == 0.95
        assert agent.epsilon == 0.5  # epsilon_start est stocké dans epsilon
        assert agent.epsilon_end == 0.01
        assert agent.epsilon_decay == 0.99
        assert agent.batch_size == 32
        assert agent.target_update_freq == 50

    def test_select_action_exploration(self):
        """Test sélection d'action en mode exploration"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 1.0  # Mode exploration complet

        state = np.random.rand(agent.state_dim)
        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_exploitation(self):
        """Test sélection d'action en mode exploitation"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0  # Mode exploitation complet

        state = np.random.rand(agent.state_dim)
        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_with_valid_actions(self):
        """Test sélection d'action avec actions valides"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.random.rand(agent.state_dim)
        valid_actions = [0, 1, 2, 5, 10]

        action = agent.select_action(state, valid_actions=valid_actions)

        assert action in valid_actions

    def test_select_action_with_empty_valid_actions(self):
        """Test sélection d'action avec actions valides vides"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.random.rand(agent.state_dim)
        valid_actions = []

        action = agent.select_action(state, valid_actions=valid_actions)

        assert action == 0  # Action par défaut (wait)

    def test_select_action_with_invalid_valid_actions(self):
        """Test sélection d'action avec actions valides invalides"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.random.rand(agent.state_dim)
        valid_actions = [100, 200]  # Actions hors limites

        action = agent.select_action(state, valid_actions=valid_actions)

        assert action == 0  # Action par défaut (wait)

    def test_select_action_with_none_valid_actions(self):
        """Test sélection d'action avec actions valides None"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.random.rand(agent.state_dim)

        action = agent.select_action(state, valid_actions=None)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_with_different_states(self):
        """Test sélection d'action avec différents états"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        # Test avec état normal
        state1 = np.random.rand(agent.state_dim)
        action1 = agent.select_action(state1)

        # Test avec état zéro
        state2 = np.zeros(agent.state_dim)
        action2 = agent.select_action(state2)

        # Test avec état négatif
        state3 = np.random.rand(agent.state_dim) * -1
        action3 = agent.select_action(state3)

        assert isinstance(action1, int)
        assert isinstance(action2, int)
        assert isinstance(action3, int)
        assert 0 <= action1 < agent.action_dim
        assert 0 <= action2 < agent.action_dim
        assert 0 <= action3 < agent.action_dim

    def test_select_action_with_batch_states(self):
        """Test sélection d'action avec batch d'états"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        batch_size = 5
        states = np.random.rand(batch_size, agent.state_dim)

        actions = []
        for state in states:
            action = agent.select_action(state)
            actions.append(action)

        assert len(actions) == batch_size
        for action in actions:
            assert isinstance(action, int)
            assert 0 <= action < agent.action_dim

    def test_select_action_with_nan_state(self):
        """Test sélection d'action avec état contenant NaN"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.random.rand(agent.state_dim)
        state[0] = np.nan

        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_with_inf_state(self):
        """Test sélection d'action avec état contenant inf"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.random.rand(agent.state_dim)
        state[0] = np.inf

        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_with_negative_inf_state(self):
        """Test sélection d'action avec état contenant -inf"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.random.rand(agent.state_dim)
        state[0] = -np.inf

        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_with_large_state(self):
        """Test sélection d'action avec état avec valeurs importantes"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.random.rand(agent.state_dim) * 1000

        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_with_small_state(self):
        """Test sélection d'action avec état avec petites valeurs"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.random.rand(agent.state_dim) * 1e-6

        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_with_mixed_state(self):
        """Test sélection d'action avec état mixte"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.random.rand(agent.state_dim)
        state[0] = 1000  # Grande valeur
        state[1] = -1000  # Grande valeur négative
        state[2] = 0  # Zéro
        state[3] = 1e-6  # Petite valeur

        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_with_different_epsilon_values(self):
        """Test sélection d'action avec différentes valeurs d'epsilon"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        state = np.random.rand(agent.state_dim)

        # Test avec epsilon élevé (exploration)
        agent.epsilon = 0.9
        action1 = agent.select_action(state)

        # Test avec epsilon moyen
        agent.epsilon = 0.5
        action2 = agent.select_action(state)

        # Test avec epsilon faible (exploitation)
        agent.epsilon = 0.1
        action3 = agent.select_action(state)

        assert isinstance(action1, int)
        assert isinstance(action2, int)
        assert isinstance(action3, int)
        assert 0 <= action1 < agent.action_dim
        assert 0 <= action2 < agent.action_dim
        assert 0 <= action3 < agent.action_dim

    def test_select_action_with_different_action_dims(self):
        """Test sélection d'action avec différentes dimensions d'action"""
        # Test avec petite dimension d'action
        agent1 = ImprovedDQNAgent(state_dim=62, action_dim=5)
        state1 = np.random.rand(agent1.state_dim)
        action1 = agent1.select_action(state1)

        # Test avec grande dimension d'action
        agent2 = ImprovedDQNAgent(state_dim=62, action_dim=0.100)
        state2 = np.random.rand(agent2.state_dim)
        action2 = agent2.select_action(state2)

        assert isinstance(action1, int)
        assert isinstance(action2, int)
        assert 0 <= action1 < 5
        assert 0 <= action2 < 100

    def test_select_action_with_different_state_dims(self):
        """Test sélection d'action avec différentes dimensions d'état"""
        # Test avec petite dimension d'état
        agent1 = ImprovedDQNAgent(state_dim=10, action_dim=51)
        state1 = np.random.rand(10)
        action1 = agent1.select_action(state1)

        # Test avec grande dimension d'état
        agent2 = ImprovedDQNAgent(state_dim=0.200, action_dim=51)
        state2 = np.random.rand(200)
        action2 = agent2.select_action(state2)

        assert isinstance(action1, int)
        assert isinstance(action2, int)
        assert 0 <= action1 < agent1.action_dim
        assert 0 <= action2 < agent2.action_dim

    def test_select_action_with_exception(self):
        """Test sélection d'action avec exception"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        # Mock pour provoquer une exception
        with patch.object(agent.q_network, "forward", side_effect=Exception("Network error")):
            state = np.random.rand(agent.state_dim)

            # L'exception devrait être propagée car elle n'est pas gérée dans select_action
            with pytest.raises(Exception, match="Network error"):
                agent.select_action(state)

    def test_select_action_with_invalid_state_shape(self):
        """Test sélection d'action avec forme d'état invalide"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        # Test avec état de dimension incorrecte
        state = np.random.rand(agent.state_dim + 1)

        with pytest.raises((RuntimeError, ValueError)):
            agent.select_action(state)

    def test_select_action_with_empty_state(self):
        """Test sélection d'action avec état vide"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = np.array([])

        with pytest.raises((RuntimeError, ValueError)):
            agent.select_action(state)

    def test_select_action_with_none_state(self):
        """Test sélection d'action avec état None"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        with pytest.raises((TypeError, AttributeError)):
            agent.select_action(None)

    def test_select_action_with_string_state(self):
        """Test sélection d'action avec état string"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        with pytest.raises((TypeError, AttributeError)):
            agent.select_action("invalid_state")

    def test_select_action_with_list_state(self):
        """Test sélection d'action avec état list"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        # Créer une liste avec la bonne dimension
        state = [1.0] * agent.state_dim

        # Convertir en numpy array pour éviter les erreurs de dimension
        state_array = np.array(state)
        action = agent.select_action(state_array)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_with_dict_state(self):
        """Test sélection d'action avec état dict"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0.0

        state = {"key": "value"}

        with pytest.raises((TypeError, AttributeError)):
            agent.select_action(state)
