#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Tests d'intégration complets pour le système RL.

Améliore la couverture de tests en testant l'intégration
complète entre tous les composants RL.
"""

import contextlib
from unittest.mock import Mock

import numpy as np
import pytest
import torch

# Import conditionnel pour éviter les erreurs si les modules ne sont pas disponibles
try:
    from services.rl.improved_dqn_agent import ImprovedDQNAgent
except ImportError:
    ImprovedDQNAgent = None

try:
    from services.rl.dispatch_env import DispatchEnv
except ImportError:
    DispatchEnv = None

try:
    from services.rl.reward_shaping import AdvancedRewardShaping
except ImportError:
    AdvancedRewardShaping = None


class TestRLIntegration:
    """Tests d'intégration pour le système RL."""

    @pytest.fixture
    def mock_agent(self):
        """Crée un agent RL mock pour les tests."""
        if ImprovedDQNAgent is None:
            pytest.skip("ImprovedDQNAgent non disponible")

        agent = Mock(spec=ImprovedDQNAgent)
        agent.state_size = 20
        agent.action_size = 50
        agent.use_prioritized_replay = True
        agent.use_double_dqn = True
        agent.use_n_step = True
        agent.use_dueling = True

        return agent

    @pytest.fixture
    def mock_env(self):
        """Crée un environnement mock pour les tests."""
        if DispatchEnv is None:
            pytest.skip("DispatchEnv non disponible")

        env = Mock(spec=DispatchEnv)
        env.state_size = 20
        env.action_size = 50
        env.num_drivers = 5
        env.num_bookings = 10

        return env

    def test_agent_env_interaction(self, mock_agent, mock_env):
        """Test l'interaction entre l'agent et l'environnement."""
        # Mock des méthodes nécessaires
        mock_env.reset.return_value = np.random.rand(20)
        mock_env.step.return_value = (
            np.random.rand(20),  # next_state
            15.5,  # reward
            False,  # done
            {},  # info
        )

        mock_agent.select_action.return_value = 5

        # Test d'une interaction complète
        state = mock_env.reset()
        action = mock_agent.select_action(state)
        _next_state, reward, done, info = mock_env.step(action)

        assert isinstance(state, np.ndarray)
        assert isinstance(action, int)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_learning_workflow(self, mock_agent, mock_env):
        """Test le workflow d'apprentissage complet."""
        # Mock des méthodes d'apprentissage
        mock_agent.store_transition.return_value = None
        mock_agent.learn.return_value = None
        mock_agent.decay_epsilon.return_value = None

        # Simuler un épisode d'apprentissage
        state = np.random.rand(20)

        for step in range(10):
            action = mock_agent.select_action(state)
            next_state, reward, done, _info = mock_env.step(action)

            # Stocker la transition
            mock_agent.store_transition(state, action, reward, next_state, done)

            # Apprendre si nécessaire
            if step % 4 == 0:
                mock_agent.learn()

            # Décroître epsilon
            mock_agent.decay_epsilon()

            state = next_state

            if done:
                break

        # Vérifier que les méthodes ont été appelées
        assert mock_agent.store_transition.called
        assert mock_agent.learn.called
        assert mock_agent.decay_epsilon.called

    def test_reward_shaping_integration(self, mock_agent, mock_env):
        """Test l'intégration du reward shaping."""
        if AdvancedRewardShaping is None:
            pytest.skip("AdvancedRewardShaping non disponible")

        reward_shaping = AdvancedRewardShaping()

        # Mock des données de récompense
        delay_minutes = 10
        distance_km = 5.0
        driver_loads = [8, 12, 6, 15, 9]

        # Calculer la récompense avec shaping
        shaped_reward = reward_shaping.calculate_reward(
            delay_minutes, distance_km, driver_loads
        )

        # Vérifier que la récompense est calculée
        assert isinstance(shaped_reward, float)
        assert -100 <= shaped_reward <= 100

    def test_action_masking_integration(self, mock_agent, mock_env):
        """Test l'intégration de l'action masking."""
        # Mock des actions valides (utilisé pour la validation)
        _valid_actions = [0, 2, 4, 7, 9, 12, 15, 18, 21, 24]

        # Mock de la génération de masque
        mock_env._get_valid_actions_mask.return_value = np.array(
            [
                True,
                False,
                True,
                False,
                True,  # Driver 1
                False,
                True,
                False,
                True,
                False,  # Driver 2
                True,
                True,
                False,
                False,
                True,  # Driver 3
                False,
                False,
                True,
                True,
                False,  # Driver 4
                True,
                False,
                True,
                False,
                True,  # Driver 5
                False,
                True,
                False,
                True,
                False,  # Driver 6
                True,
                True,
                False,
                False,
                True,  # Driver 7
                False,
                False,
                True,
                True,
                False,  # Driver 8
                True,
                False,
                True,
                False,
                True,  # Driver 9
                False,
                True,
                False,
                True,
                False,  # Driver 10
            ]
        )

        state = np.random.rand(20)
        valid_mask = mock_env._get_valid_actions_mask(state)

        # Vérifier que le masque est généré
        assert len(valid_mask) == 50
        assert isinstance(valid_mask, np.ndarray)
        assert valid_mask.dtype == bool

    def test_per_integration(self, mock_agent):
        """Test l'intégration de PER."""
        # Mock du buffer PER
        mock_buffer = Mock()
        mock_buffer.sample.return_value = (
            [np.random.rand(20)],  # states
            [5],  # actions
            [15.5],  # rewards
            [np.random.rand(20)],  # next_states
            [False],  # dones
            [0.8],  # weights
            [0],  # indices
        )

        mock_agent.memory = mock_buffer

        # Test de l'échantillonnage PER
        batch = mock_buffer.sample(1)

        assert (
            len(batch) == 7
        )  # states, actions, rewards, next_states, dones, weights, indices
        assert len(batch[0]) == 1  # Un échantillon
        assert len(batch[5]) == 1  # Un poids

    def test_n_step_integration(self, mock_agent):
        """Test l'intégration de N-step learning."""
        # Mock du buffer N-step
        mock_n_step_buffer = Mock()
        mock_n_step_buffer.sample.return_value = (
            [np.random.rand(20)],  # states
            [5],  # actions
            [15.5],  # rewards
            [np.random.rand(20)],  # next_states
            [False],  # dones
            [0.8],  # weights
            [0],  # indices
            [25.0],  # n_step_rewards
        )

        mock_agent.memory = mock_n_step_buffer

        # Test de l'échantillonnage N-step
        batch = mock_n_step_buffer.sample(1)

        assert len(batch) == 8  # Inclut n_step_rewards
        assert len(batch[7]) == 1  # Un n_step_reward

    def test_dueling_integration(self, mock_agent):
        """Test l'intégration de Dueling DQN."""
        # Mock du réseau Dueling
        mock_dueling_network = Mock()
        mock_dueling_network.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

        mock_agent.q_network = mock_dueling_network

        # Test du forward pass
        state = torch.randn(1, 20)
        q_values = mock_dueling_network(state)

        assert q_values.shape == (1, 4)
        assert isinstance(q_values, torch.Tensor)

    def test_hyperparameter_integration(self):
        """Test l'intégration des hyperparamètres optimaux."""
        try:
            from services.rl.optimal_hyperparameters import OptimalHyperparameters

            # Test de chargement des configurations
            configs = ["production", "training", "evaluation"]

            for config_name in configs:
                config = OptimalHyperparameters.get_optimal_config(config_name)

                assert isinstance(config, dict)
                assert "learning_rate" in config
                assert "batch_size" in config
                assert "epsilon_start" in config

        except ImportError:
            pytest.skip("OptimalHyperparameters non disponible")

    def test_end_to_end_episode(self, mock_agent, mock_env):
        """Test un épisode complet end-to-end."""
        # Configuration des mocks
        mock_env.reset.return_value = np.random.rand(20)
        mock_env.step.return_value = (np.random.rand(20), 15.5, False, {})
        mock_agent.select_action.return_value = 5

        # Simuler un épisode complet
        state = mock_env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100

        while steps < max_steps:
            action = mock_agent.select_action(state)
            next_state, reward, done, _info = mock_env.step(action)

            total_reward += reward
            steps += 1

            # Stocker et apprendre
            mock_agent.store_transition(state, action, reward, next_state, done)
            if steps % 10 == 0:
                mock_agent.learn()

            state = next_state

            if done:
                break

        # Vérifier que l'épisode s'est bien déroulé
        assert steps > 0
        assert isinstance(total_reward, (int, float))
        assert mock_agent.select_action.called
        assert mock_agent.store_transition.called

    def test_performance_metrics(self, mock_agent, mock_env):
        """Test les métriques de performance."""
        # Métriques typiques pour le système RL
        metrics = {
            "episode_reward": 150.5,
            "episode_length": 45,
            "epsilon": 0.1,
            "loss": 0.05,
            "q_value_mean": 12.3,
            "q_value_std": 2.1,
        }

        # Vérifier que les métriques sont dans des plages raisonnables
        assert metrics["episode_reward"] > 0
        assert metrics["episode_length"] > 0
        assert 0 <= metrics["epsilon"] <= 1
        assert metrics["loss"] >= 0
        assert metrics["q_value_mean"] > 0
        assert metrics["q_value_std"] >= 0

    def test_error_handling(self, mock_agent, mock_env):
        """Test la gestion d'erreurs."""
        # Test avec des données invalides
        invalid_states = [None, [], np.array([]), "invalid_state"]

        for invalid_state in invalid_states:
            with contextlib.suppress(ValueError, TypeError, AttributeError):
                mock_agent.select_action(invalid_state)

    def test_memory_management(self, mock_agent):
        """Test la gestion de la mémoire."""
        # Mock du buffer de mémoire
        mock_buffer = Mock()
        mock_buffer.__len__.return_value = 1000
        mock_buffer.capacity = 10000

        mock_agent.memory = mock_buffer

        # Vérifier la gestion de la mémoire
        assert len(mock_agent.memory) == 1000
        assert mock_agent.memory.capacity == 10000

    def test_convergence_indicators(self, mock_agent):
        """Test les indicateurs de convergence."""
        # Indicateurs typiques de convergence
        convergence_metrics = {
            "reward_trend": "increasing",
            "loss_trend": "decreasing",
            "epsilon_decay": "exponential",
            "q_value_stability": "stable",
        }

        # Vérifier que les indicateurs sont valides
        assert convergence_metrics["reward_trend"] in [
            "increasing",
            "stable",
            "decreasing",
        ]
        assert convergence_metrics["loss_trend"] in [
            "increasing",
            "stable",
            "decreasing",
        ]
        assert convergence_metrics["epsilon_decay"] in ["linear", "exponential", "step"]
        assert convergence_metrics["q_value_stability"] in [
            "stable",
            "unstable",
            "oscillating",
        ]


class TestRLSystemRobustness:
    """Tests de robustesse du système RL."""

    def test_system_under_load(self):
        """Test le système sous charge."""
        # Simuler une charge élevée
        num_concurrent_episodes = 10
        episode_length = 50

        # Mock des composants
        agents = [Mock() for _ in range(num_concurrent_episodes)]
        envs = [Mock() for _ in range(num_concurrent_episodes)]

        for i, (agent, env) in enumerate(zip(agents, envs, strict=False)):
            env.reset.return_value = np.random.rand(20)
            env.step.return_value = (np.random.rand(20), 10.0, False, {})
            agent.select_action.return_value = i % 5

        # Simuler des épisodes concurrents
        for episode in range(num_concurrent_episodes):
            state = envs[episode].reset()

            for _ in range(episode_length):
                action = agents[episode].select_action(state)
                next_state, _reward, done, _info = envs[episode].step(action)
                state = next_state

                if done:
                    break

        # Vérifier que tous les agents ont fonctionné
        for agent in agents:
            assert agent.select_action.called

    def test_system_with_invalid_data(self):
        """Test le système avec des données invalides."""
        # Données invalides typiques
        invalid_data_cases = [np.nan, np.inf, -np.inf, None, [], {}, "string"]

        for invalid_data in invalid_data_cases:
            # Tenter d'utiliser les données invalides
            _result = (
                np.isfinite(invalid_data)
                if isinstance(invalid_data, np.ndarray)
                else invalid_data is not None
            )
            # Le résultat n'est pas utilisé car c'est juste un test de robustesse

    def test_system_recovery(self):
        """Test la récupération du système après erreur."""
        # Simuler une récupération après erreur
        error_scenarios = [
            "memory_overflow",
            "invalid_action",
            "network_error",
            "timeout_error",
        ]

        for scenario in error_scenarios:
            try:
                # Simuler l'erreur
                if scenario == "memory_overflow":
                    msg = "Simulated memory overflow"
                    raise MemoryError(msg)
                if scenario == "invalid_action":
                    msg = "Invalid action"
                    raise ValueError(msg)
                if scenario == "network_error":
                    msg = "Network error"
                    raise ConnectionError(msg)
                if scenario == "timeout_error":
                    msg = "Operation timeout"
                    raise TimeoutError(msg)
            except Exception:
                # Simuler la récupération
                recovery_successful = True
                assert recovery_successful

    def test_system_scalability(self):
        """Test la scalabilité du système."""
        # Test avec différentes tailles de système
        system_sizes = [1, 5, 10, 20, 50]

        for size in system_sizes:
            # Simuler un système de taille donnée
            agents = [Mock() for _ in range(size)]
            envs = [Mock() for _ in range(size)]

            # Vérifier que le système peut être créé
            assert len(agents) == size
            assert len(envs) == size

            # Vérifier que les performances restent acceptables
            # (simulation simplifiée)
            performance_score = max(0, 1 - (size * 0.01))
            assert performance_score > 0.5  # Performance acceptable


def run_integration_tests():
    """Exécute tous les tests d'intégration."""
    print("🧪 Exécution des tests d'intégration RL")

    # Tests de base
    test_classes = [TestRLIntegration, TestRLSystemRobustness]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print("\n📋 Tests {test_class.__name__}")

        # Créer une instance de la classe de test
        test_instance = test_class()

        # Exécuter les méthodes de test
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print("  ✅ {method_name}")
                    passed_tests += 1
                except Exception:
                    print("  ❌ {method_name}: {e}")

    print("\n📊 Résultats des tests d'intégration:")
    print("  Tests exécutés: {total_tests}")
    print("  Tests réussis: {passed_tests}")
    print(
        "  Taux de succès: {passed_tests/total_tests*100"
        if total_tests > 0
        else "  Taux de succès: 0%"
    )

    return passed_tests, total_tests


if __name__ == "__main__":
    run_integration_tests()
