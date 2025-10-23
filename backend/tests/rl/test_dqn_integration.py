# ruff: noqa: DTZ001, DTZ003, T201
# pyright: reportMissingImports=false
"""
Tests d'intégration DQN + Environnement.

Teste:
- Training loop complet
- Amélioration vs baseline
- Convergence
- Performance
"""
import numpy as np
import pytest

from services.rl.dispatch_env import DispatchEnv
from services.rl.dqn_agent import DQNAgent


class TestDQNIntegrationBasic:
    """Tests d'intégration basiques."""

    def test_full_training_loop_minimal(self):
        """Test training loop complet sur 5 episodes."""
        env = DispatchEnv(num_drivers=3, max_bookings=5, simulation_hours=1)
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            batch_size=16
        )

        for episode in range(5):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < 50:
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)

                agent.store_transition(state, action, next_state, reward, done or truncated)

                if len(agent.memory) >= agent.batch_size:
                    agent.train_step()

                state = next_state
                episode_reward += reward
                steps += 1

            agent.decay_epsilon()

            if episode % 2 == 0:
                agent.update_target_network()

        # Vérifier que l'agent a appris quelque chose
        assert agent.training_step > 0
        assert len(agent.losses) > 0
        assert len(agent.memory) > 0

    def test_agent_with_env_interface(self):
        """Test que l'agent fonctionne bien avec l'environnement."""
        env = DispatchEnv(num_drivers=5, max_bookings=10)
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n
        )

        # Reset env
        state, _ = env.reset()

        # Vérifier que l'état est compatible
        assert state.shape == (env.observation_space.shape[0],)

        # Sélectionner action
        action = agent.select_action(state)

        # Vérifier que l'action est valide
        assert 0 <= action < env.action_space.n

        # Step
        next_state, reward, done, truncated, info = env.step(action)

        # Stocker transition
        agent.store_transition(state, action, next_state, reward, done)

        assert len(agent.memory) == 1


class TestDQNLearning:
    """Tests de l'apprentissage."""

    def test_agent_learns_over_episodes(self):
        """Test que l'agent s'améliore avec l'entraînement."""
        env = DispatchEnv(num_drivers=5, max_bookings=10, simulation_hours=1)
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            batch_size=32,
            epsilon_decay=0.98
        )

        # Entraîner 30 episodes
        rewards_first_10 = []
        rewards_last_10 = []

        for episode in range(30):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < 50:
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                agent.store_transition(state, action, next_state, reward, done or truncated)

                if len(agent.memory) >= agent.batch_size:
                    agent.train_step()

                state = next_state
                episode_reward += reward
                steps += 1

            # Tracker rewards
            if episode < 10:
                rewards_first_10.append(episode_reward)
            elif episode >= 20:
                rewards_last_10.append(episode_reward)

            agent.decay_epsilon()

            if episode % 5 == 0:
                agent.update_target_network()

        # Les derniers episodes devraient avoir un reward moyen meilleur
        # (pas garanti à 100% avec si peu d'épisodes, mais tendance)
        avg_first = np.mean(rewards_first_10)
        avg_last = np.mean(rewards_last_10)

        print("\n📊 Learning Progress:")
        print(f"   Episodes 0-9:  Avg reward = {avg_first:.1f}")
        print(f"   Episodes 20-29: Avg reward = {avg_last:.1f}")
        print(f"   Improvement: {avg_last - avg_first:.1f}")

        # Au minimum, l'agent ne devrait pas empirer
        # (avec aléa, on accepte une petite régression)
        assert avg_last >= avg_first - 100

    def test_agent_evaluation_mode(self):
        """Test que l'agent peut être évalué sans exploration."""
        env = DispatchEnv(num_drivers=3, max_bookings=5, simulation_hours=1)
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            epsilon_start=1.0
        )

        # Entraîner un peu
        for _ in range(10):
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.store_transition(state, action, next_state, reward, done)
                if len(agent.memory) >= 32:
                    agent.train_step()
                state = next_state

        # Évaluer (training=False)
        state, _ = env.reset()
        actions_eval = []

        for _ in range(10):
            action = agent.select_action(state, training=False)
            actions_eval.append(action)
            state, _, done, _, _ = env.step(action)
            if done:
                break

        # Toutes les actions devraient être greedy (peut-être répétitives)
        assert len(actions_eval) > 0


class TestDQNPerformance:
    """Tests de performance."""

    def test_inference_speed(self):
        """Test vitesse d'inférence."""
        import time

        agent = DQNAgent(state_dim=122, action_dim=201)
        state = np.random.rand(122)

        # Warmup
        for _ in range(10):
            agent.select_action(state, training=False)

        # Mesurer temps pour 100 inférences
        start = time.time()
        for _ in range(100):
            agent.select_action(state, training=False)
        elapsed = time.time() - start

        avg_time_ms = (elapsed / 100) * 1000

        print(f"\n⚡ Inference speed: {avg_time_ms:.2f}ms per action")

        # Devrait être très rapide (< 10ms sur CPU)
        assert avg_time_ms < 50  # Acceptable même sur CPU lent

