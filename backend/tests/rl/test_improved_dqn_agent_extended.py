#!/usr/bin/env python3
"""
Tests supplémentaires pour ImprovedDQNAgent - Version corrigée
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from services.rl.improved_dqn_agent import ImprovedDQNAgent


class TestImprovedDQNAgentExtended:
    """Tests étendus pour améliorer la couverture"""

    def test_init_with_torch_import_error(self):
        """Test initialisation avec erreur d'import PyTorch"""
        with patch("services.rl.improved_dqn_agent.torch", None):
            with pytest.raises(ImportError, match="PyTorch is required"):
                ImprovedDQNAgent(state_dim=62, action_dim=51)

    def test_init_with_n_step_import_error(self):
        """Test initialisation avec erreur d'import N-step"""
        with patch("services.rl.improved_dqn_agent.create_n_step_buffer", None):
            with pytest.raises(ImportError, match="N-step learning"):
                ImprovedDQNAgent(state_dim=62, action_dim=51, use_n_step=True)

    def test_init_with_dueling_import_error(self):
        """Test initialisation avec erreur d'import Dueling"""
        with patch("services.rl.improved_dqn_agent.DuelingQNetwork", None):
            with pytest.raises(ImportError, match="Dueling DQN"):
                ImprovedDQNAgent(state_dim=62, action_dim=51, use_dueling=True)

    def test_init_with_per_import_error(self):
        """Test initialisation avec erreur d'import PER"""
        with patch("services.rl.improved_dqn_agent.PrioritizedReplayBuffer", None):
            with pytest.raises(ImportError, match="Prioritized Experience Replay"):
                ImprovedDQNAgent(state_dim=62, action_dim=51, use_prioritized_replay=True)

    def test_select_action_with_rl_logger_import_error(self):
        """Test select_action avec erreur d'import RLLogger"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0

        # Mock l'import dans select_action
        with patch("builtins.__import__", side_effect=ImportError("RLLogger not available")):
            state = np.random.rand(agent.state_dim)
            action = agent.select_action(state)

            assert isinstance(action, int)
            assert 0 <= action < agent.action_dim

    def test_select_action_with_rl_logger_logging_error(self):
        """Test select_action avec erreur de logging RLLogger"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0

        # Mock RLLogger qui lève une exception lors du logging
        mock_logger = Mock()
        mock_logger.log_decision.side_effect = Exception("Logging error")

        # Mock l'import pour retourner notre mock
        def mock_import(name, *args, **kwargs):
            if name == "services.rl.rl_logger":
                mock_module = Mock()
                mock_module.get_rl_logger.return_value = mock_logger
                return mock_module
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            state = np.random.rand(agent.state_dim)
            action = agent.select_action(state)

            # L'action devrait quand même être retournée malgré l'erreur de logging
            assert isinstance(action, int)
            assert 0 <= action < agent.action_dim

    def test_select_action_exploration_branch(self):
        """Test branche exploration dans select_action"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 1  # Force exploration

        state = np.random.rand(agent.state_dim)
        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_exploitation_branch(self):
        """Test branche exploitation dans select_action"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0  # Force exploitation

        state = np.random.rand(agent.state_dim)
        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < agent.action_dim

    def test_select_action_with_valid_actions_exploration(self):
        """Test select_action avec valid_actions en mode exploration"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 1  # Force exploration

        state = np.random.rand(agent.state_dim)
        valid_actions = [0, 1, 2, 5, 10]

        action = agent.select_action(state, valid_actions=valid_actions)

        assert action in valid_actions

    def test_select_action_with_valid_actions_exploitation(self):
        """Test select_action avec valid_actions en mode exploitation"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0  # Force exploitation

        state = np.random.rand(agent.state_dim)
        valid_actions = [0, 1, 2, 5, 10]

        action = agent.select_action(state, valid_actions=valid_actions)

        assert action in valid_actions

    def test_select_action_with_empty_valid_actions_fallback(self):
        """Test select_action avec valid_actions vide - fallback"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        agent.epsilon = 0

        state = np.random.rand(agent.state_dim)
        valid_actions = []

        action = agent.select_action(state, valid_actions=valid_actions)

        assert action == 0  # Fallback vers action 0

    def test_store_transition_basic(self):
        """Test méthode store_transition"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51, use_n_step=False, use_prioritized_replay=False)

        state = np.random.rand(agent.state_dim)
        action = 1
        reward = 10
        next_state = np.random.rand(agent.state_dim)
        done = False

        # Devrait fonctionner sans erreur
        agent.store_transition(state, action, reward, next_state, done)

        # Vérifier que la transition a été stockée
        assert len(agent.memory) > 0

    def test_store_transition_with_n_step(self):
        """Test méthode store_transition avec N-step"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51, use_n_step=True)

        state = np.random.rand(agent.state_dim)
        action = 1
        reward = 10
        next_state = np.random.rand(agent.state_dim)
        done = False

        # Devrait fonctionner sans erreur
        agent.store_transition(state, action, reward, next_state, done)

        # Vérifier que la transition a été stockée
        assert len(agent.memory) > 0

    def test_store_transition_with_per(self):
        """Test méthode store_transition avec PER"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51, use_prioritized_replay=True)

        state = np.random.rand(agent.state_dim)
        action = 1
        reward = 10
        next_state = np.random.rand(agent.state_dim)
        done = False

        # Devrait fonctionner sans erreur
        agent.store_transition(state, action, reward, next_state, done)

        # Vérifier que la transition a été stockée
        assert len(agent.memory) > 0

    def test_learn_basic(self):
        """Test méthode learn basique"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)

        # Ajouter quelques transitions pour avoir assez de données
        for _ in range(100):
            state = np.random.rand(agent.state_dim)
            action = np.random.randint(0, agent.action_dim)
            reward = np.random.randn()
            next_state = np.random.rand(agent.state_dim)
            done = np.random.choice([True, False])
            agent.store_transition(state, action, reward, next_state, done)

        # Devrait fonctionner sans erreur
        loss = agent.learn()
        assert isinstance(loss, float)

    def test_learn_with_insufficient_data(self):
        """Test méthode learn avec données insuffisantes"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)

        # Ajouter seulement quelques transitions (pas assez pour un batch)
        for _ in range(10):
            state = np.random.rand(agent.state_dim)
            action = np.random.randint(0, agent.action_dim)
            reward = np.random.randn()
            next_state = np.random.rand(agent.state_dim)
            done = np.random.choice([True, False])
            agent.store_transition(state, action, reward, next_state, done)

        # Devrait fonctionner sans erreur (pas d'apprentissage)
        loss = agent.learn()
        assert loss == 0

    def test_soft_update_target_network(self):
        """Test méthode _soft_update_target_network"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)

        # Devrait fonctionner sans erreur
        agent._soft_update_target_network()

    def test_decay_epsilon(self):
        """Test méthode decay_epsilon"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)
        initial_epsilon = agent.epsilon

        # Devrait fonctionner sans erreur
        agent.decay_epsilon()

        # Epsilon devrait avoir diminué
        assert agent.epsilon <= initial_epsilon

    def test_save_model(self):
        """Test méthode save"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)

        # Mock torch.save pour éviter de créer un vrai fichier
        with patch("torch.save") as mock_save:
            agent.save("test_model.pth")
            mock_save.assert_called_once()

    def test_load_model(self):
        """Test méthode load"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51)

        # Mock torch.load pour éviter de charger un vrai fichier
        with patch("torch.load") as mock_load:
            mock_load.return_value = {
                "q_network_state_dict": {},
                "target_network_state_dict": {},
                "optimizer_state_dict": {},
                "epsilon": 0.1,
                "training_step": 0,
                "episode_count": 0,
                "losses": [],
                "config": {}
            }
            agent.load("test_model.pth")
            mock_load.assert_called_once()

    def test_learn_with_n_step_buffer(self):
        """Test méthode learn avec buffer N-step"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51, use_n_step=True)

        # Ajouter quelques transitions
        for _ in range(100):
            state = np.random.rand(agent.state_dim)
            action = np.random.randint(0, agent.action_dim)
            reward = np.random.randn()
            next_state = np.random.rand(agent.state_dim)
            done = np.random.choice([True, False])
            agent.store_transition(state, action, reward, next_state, done)

        # Devrait fonctionner sans erreur
        loss = agent.learn()
        assert isinstance(loss, float)

    def test_learn_with_per_buffer(self):
        """Test méthode learn avec buffer PER"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51, use_prioritized_replay=True)

        # Ajouter quelques transitions
        for _ in range(100):
            state = np.random.rand(agent.state_dim)
            action = np.random.randint(0, agent.action_dim)
            reward = np.random.randn()
            next_state = np.random.rand(agent.state_dim)
            done = np.random.choice([True, False])
            agent.store_transition(state, action, reward, next_state, done)

        # Devrait fonctionner sans erreur
        loss = agent.learn()
        assert isinstance(loss, float)

    def test_learn_with_double_dqn(self):
        """Test méthode learn avec Double DQN"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51, use_double_dqn=True)

        # Ajouter quelques transitions
        for _ in range(100):
            state = np.random.rand(agent.state_dim)
            action = np.random.randint(0, agent.action_dim)
            reward = np.random.randn()
            next_state = np.random.rand(agent.state_dim)
            done = np.random.choice([True, False])
            agent.store_transition(state, action, reward, next_state, done)

        # Devrait fonctionner sans erreur
        loss = agent.learn()
        assert isinstance(loss, float)

    def test_learn_with_standard_dqn(self):
        """Test méthode learn avec DQN standard"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51, use_double_dqn=False)

        # Ajouter quelques transitions
        for _ in range(100):
            state = np.random.rand(agent.state_dim)
            action = np.random.randint(0, agent.action_dim)
            reward = np.random.randn()
            next_state = np.random.rand(agent.state_dim)
            done = np.random.choice([True, False])
            agent.store_transition(state, action, reward, next_state, done)

        # Devrait fonctionner sans erreur
        loss = agent.learn()
        assert isinstance(loss, float)

    def test_learn_with_target_update(self):
        """Test méthode learn avec mise à jour du réseau cible"""
        agent = ImprovedDQNAgent(state_dim=62, action_dim=51, target_update_freq=1)

        # Ajouter quelques transitions
        for _ in range(100):
            state = np.random.rand(agent.state_dim)
            action = np.random.randint(0, agent.action_dim)
            reward = np.random.randn()
            next_state = np.random.rand(agent.state_dim)
            done = np.random.choice([True, False])
            agent.store_transition(state, action, reward, next_state, done)

        # Devrait fonctionner sans erreur
        loss = agent.learn()
        assert isinstance(loss, float)
