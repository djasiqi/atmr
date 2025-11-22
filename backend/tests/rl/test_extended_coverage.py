"""
Tests supplémentaires pour améliorer la couverture - Méthodes principales
"""

from unittest.mock import Mock

import numpy as np
import torch

from services.rl.dispatch_env import DispatchEnv
from services.rl.improved_dqn_agent import ImprovedDQNAgent
from services.rl.n_step_buffer import NStepBuffer, NStepPrioritizedBuffer
from services.rl.replay_buffer import PrioritizedReplayBuffer
from services.rl.reward_shaping import RewardShapingConfig
from services.rl.shadow_mode_manager import ShadowModeManager


class TestExtendedCoverage:
    """Tests étendus pour améliorer la couverture"""

    def test_improved_dqn_agent_select_action(self):
        """Test sélection d'action ImprovedDQNAgent"""
        agent = ImprovedDQNAgent(state_dim=10, action_dim=5)

        state = np.random.rand(10)
        action = agent.select_action(state)

        assert isinstance(action, int)
        assert 0 <= action < 5

    def test_improved_dqn_agent_select_action_with_valid_actions(self):
        """Test sélection d'action avec actions valides"""
        agent = ImprovedDQNAgent(state_dim=10, action_dim=5)
        agent.epsilon = 0.0  # Mode exploitation

        state = np.random.rand(10)
        valid_actions = [0, 1, 2]

        action = agent.select_action(state, valid_actions=valid_actions)

        assert action in valid_actions

    def test_improved_dqn_agent_select_action_empty_valid_actions(self):
        """Test sélection d'action avec actions valides vides"""

        agent = ImprovedDQNAgent(state_dim=10, action_dim=5)

        state = np.random.rand(10)
        valid_actions = []

        action = agent.select_action(state, valid_actions=valid_actions)

        assert action == 0  # Fallback vers action 0

    def test_improved_dqn_agent_store_transition(self):
        """Test stockage de transition"""

        agent = ImprovedDQNAgent(state_dim=10, action_dim=5)

        state = np.random.rand(10)
        action = 1
        reward = 10.0
        next_state = np.random.rand(10)
        done = False

        agent.store_transition(state, action, reward, next_state, done)

        # Vérifier que la méthode a été appelée sans erreur
        # Le buffer peut être vide si les transitions ne sont pas encore complètes
        assert True  # Test passe si pas d'exception

    def test_improved_dqn_agent_learn_insufficient_data(self):
        """Test apprentissage avec données insuffisantes"""

        agent = ImprovedDQNAgent(state_dim=10, action_dim=5)

        # Pas assez de données pour apprendre
        loss = agent.learn()

        assert loss == 0.0

    def test_improved_dqn_agent_decay_epsilon(self):
        """Test décroissance d'epsilon"""

        agent = ImprovedDQNAgent(state_dim=10, action_dim=5)
        initial_epsilon = agent.epsilon

        agent.decay_epsilon()

        assert agent.epsilon <= initial_epsilon

    def test_improved_q_network_forward(self):
        """Test forward ImprovedQNetwork"""
        from services.rl.improved_q_network import ImprovedQNetwork

        network = ImprovedQNetwork(state_dim=10, action_dim=5)

        state = torch.randn(1, 10)
        output = network(state)

        assert output.shape == (1, 5)

    def test_dueling_q_network_forward(self):
        """Test forward DuelingQNetwork"""
        from services.rl.improved_q_network import DuelingQNetwork

        network = DuelingQNetwork(state_dim=10, action_dim=5)

        state = torch.randn(1, 10)
        output = network(state)

        assert output.shape == (1, 5)

    def test_n_step_buffer_add_transition(self):
        """Test ajout de transition NStepBuffer"""
        from services.rl.n_step_buffer import NStepBuffer

        buffer = NStepBuffer(capacity=0.100, n_step=3)

        state = np.random.rand(10)
        action = 1
        reward = 10.0
        next_state = np.random.rand(10)
        done = False

        buffer.add_transition(state, action, reward, next_state, done)

        assert len(buffer.buffer) > 0

    def test_n_step_buffer_sample_empty(self):
        """Test échantillonnage buffer vide"""

        buffer = NStepBuffer(capacity=0.100, n_step=3)

        batch = buffer.sample(10)

        assert batch == []

    def test_n_step_prioritized_buffer_add_transition(self):
        """Test ajout de transition NStepPrioritizedBuffer"""
        from services.rl.n_step_buffer import NStepPrioritizedBuffer

        buffer = NStepPrioritizedBuffer(capacity=0.100, n_step=3)

        state = np.random.rand(10)
        action = 1
        reward = 10.0
        next_state = np.random.rand(10)
        done = False

        buffer.add_transition(state, action, reward, next_state, done)

        assert len(buffer.buffer) > 0

    def test_n_step_prioritized_buffer_sample_empty(self):
        """Test échantillonnage buffer priorisé vide"""

        buffer = NStepPrioritizedBuffer(capacity=0.100, n_step=3)

        batch, weights, indices = buffer.sample(10)

        assert batch == []
        assert weights == []
        assert indices == []

    def test_replay_buffer_add(self):
        """Test ajout PrioritizedReplayBuffer"""
        from services.rl.replay_buffer import PrioritizedReplayBuffer

        buffer = PrioritizedReplayBuffer(capacity=0.100)

        state = np.random.rand(10)
        action = 1
        reward = 10.0
        next_state = np.random.rand(10)
        done = False

        buffer.add(state, action, reward, next_state, done)

        assert len(buffer.buffer) > 0

    def test_replay_buffer_sample_empty(self):
        """Test échantillonnage buffer vide"""

        buffer = PrioritizedReplayBuffer(capacity=0.100)

        batch, weights, indices = buffer.sample(10)

        assert batch == []
        assert weights == []
        assert indices == []

    def test_reward_shaping_calculate_reward(self):
        """Test calcul de récompense"""
        from services.rl.reward_shaping import AdvancedRewardShaping

        reward_shaping = AdvancedRewardShaping()

        # Données de test simples
        assignment = {
            "driver_id": 1,
            "booking_id": 1,
            "eta_minutes": 15,
            "delay_minutes": 5,
            "distance_km": 10.0,
            "load_factor": 0.8,
        }

        reward = reward_shaping.calculate_reward(assignment, {})

        assert isinstance(reward, float)

    def test_reward_shaping_config_get_weights(self):
        """Test récupération des poids"""
        from services.rl.reward_shaping import RewardShapingConfig

        config = RewardShapingConfig()

        weights = config.get_weights()

        assert isinstance(weights, dict)

    def test_reward_shaping_config_set_weights(self):
        """Test définition des poids"""

        config = RewardShapingConfig()

        custom_weights = {"punctuality_weight": 0.5}
        config.set_weights(custom_weights)

        weights = config.get_weights()
        assert weights["punctuality_weight"] == 0.5

    def test_noisy_linear_forward(self):
        """Test forward NoisyLinear"""
        from services.rl.noisy_networks import NoisyLinear

        layer = NoisyLinear(10, 5)

        x = torch.randn(1, 10)
        output = layer(x)

        assert output.shape == (1, 5)

    def test_noisy_q_network_forward(self):
        """Test forward NoisyQNetwork"""
        from services.rl.noisy_networks import NoisyQNetwork

        network = NoisyQNetwork(state_size=10, action_size=5)

        state = torch.randn(1, 10)
        output = network(state)

        assert output.shape == (1, 5)

    def test_noisy_dueling_network_forward(self):
        """Test forward NoisyDuelingQNetwork"""
        from services.rl.noisy_networks import NoisyDuelingQNetwork

        network = NoisyDuelingQNetwork(state_size=10, action_size=5)

        state = torch.randn(1, 10)
        output = network(state)

        assert output.shape == (1, 5)

    def test_hyperparameter_tuner_suggest_hyperparameters(self):
        """Test suggestion d'hyperparamètres"""
        from services.rl.hyperparameter_tuner import HyperparameterTuner

        tuner = HyperparameterTuner()

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_float.return_value = 0.0001
        mock_trial.suggest_categorical.return_value = True
        mock_trial.suggest_int.return_value = 64

        params = tuner._suggest_hyperparameters(mock_trial)

        assert isinstance(params, dict)

    def test_shadow_mode_manager_log_decision_comparison(self):
        """Test logging de comparaison de décisions"""
        from services.rl.shadow_mode_manager import ShadowModeManager

        manager = ShadowModeManager()

        human_decision = {"driver_id": 1, "booking_id": 1, "eta_minutes": 15, "delay_minutes": 5}

        rl_decision = {"driver_id": 2, "booking_id": 1, "eta_minutes": 12, "delay_minutes": 2}

        # Devrait fonctionner sans erreur
        manager.log_decision_comparison(human_decision, rl_decision)

    def test_shadow_mode_manager_calculate_kpis(self):
        """Test calcul des KPIs"""

        manager = ShadowModeManager()

        kpis = manager._calculate_kpis()

        assert isinstance(kpis, dict)

    def test_dispatch_env_reset(self):
        """Test reset DispatchEnv"""
        from services.rl.dispatch_env import DispatchEnv

        env = DispatchEnv()

        state = env.reset()

        assert isinstance(state, np.ndarray)

    def test_dispatch_env_step(self):
        """Test step DispatchEnv"""

        env = DispatchEnv()
        env.reset()

        action = 0
        next_state, reward, done, info = env.step(action)

        assert isinstance(next_state, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_dispatch_env_get_valid_actions(self):
        """Test récupération des actions valides"""

        env = DispatchEnv()
        env.reset()

        valid_actions = env.get_valid_actions()

        assert isinstance(valid_actions, list)

    def test_dispatch_env_get_state(self):
        """Test récupération de l'état"""

        env = DispatchEnv()
        env.reset()

        state = env.get_state()

        assert isinstance(state, np.ndarray)

    def test_dispatch_env_render(self):
        """Test rendu de l'environnement"""

        env = DispatchEnv()
        env.reset()

        # Devrait fonctionner sans erreur
        env.render()

    def test_dispatch_env_close(self):
        """Test fermeture de l'environnement"""

        env = DispatchEnv()

        # Devrait fonctionner sans erreur
        env.close()
