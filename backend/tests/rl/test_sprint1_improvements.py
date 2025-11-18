# pyright: reportMissingImports=false
"""
Tests unitaires pour les améliorations RL du Sprint 1.

Tests couverts:
- Prioritized Experience Replay (PER)
- Action Masking avancé
- Invariants de récompense
- Métriques baseline

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from services.rl.dispatch_env import DispatchEnv
from services.rl.improved_dqn_agent import ImprovedDQNAgent
from services.rl.replay_buffer import PrioritizedReplayBuffer
from services.rl.reward_shaping import AdvancedRewardShaping, RewardShapingConfig


class TestPrioritizedReplayBuffer:
    """Tests pour le Prioritized Experience Replay."""

    def test_per_sampling(self):
        """Test échantillonnage prioritaire."""
        buffer = PrioritizedReplayBuffer(1000, alpha=0.6, beta_start=0.4, beta_end=1.0)

        # Ajouter transitions avec priorités différentes
        high_priority_state = np.random.randn(10)
        low_priority_state = np.random.randn(10)

        buffer.add(high_priority_state, 1, 10.0, high_priority_state, False, priority=10.0)
        buffer.add(low_priority_state, 2, 1.0, low_priority_state, False, priority=1.0)

        # Échantillonner plusieurs fois
        high_priority_count = 0
        for _ in range(100):
            _batch, indices, _weights = buffer.sample(1)
            if indices[0] == 0:  # Première transition (haute priorité)
                high_priority_count += 1

        # Vérifier que haute priorité est plus souvent échantillonnée
        assert high_priority_count > 50  # Plus de 50% du temps

    def test_per_update_priorities(self):
        """Test mise à jour des priorités."""
        buffer = PrioritizedReplayBuffer(100, alpha=0.6)

        # Ajouter transition
        state = np.random.randn(10)
        buffer.add(state, 1, 5.0, state, False, priority=5.0)

        # Échantillonner
        _batch, indices, _weights = buffer.sample(1)

        # Mettre à jour priorité
        new_priorities = [10.0]  # Priorité plus élevée
        buffer.update_priorities(indices, new_priorities)

        # Vérifier que priorité a été mise à jour
        assert buffer.priorities[0] == 10.0

    def test_per_importance_sampling_weights(self):
        """Test poids d'importance sampling."""
        buffer = PrioritizedReplayBuffer(100, alpha=0.6, beta_start=0.4, beta_end=1.0)

        # Ajouter plusieurs transitions
        for i in range(10):
            state = np.random.randn(10)
            buffer.add(state, i, float(i), state, False, priority=float(i + 1))

        # Échantillonner et vérifier les poids
        _batch, _indices, weights = buffer.sample(5)

        # Les poids doivent être des valeurs positives
        assert all(w > 0 for w in weights)
        assert len(weights) == 5

    def test_per_buffer_capacity(self):
        """Test capacité du buffer."""
        buffer = PrioritizedReplayBuffer(50, alpha=0.6)

        # Ajouter plus de transitions que la capacité
        for i in range(100):
            state = np.random.randn(10)
            buffer.add(state, i, float(i), state, False, priority=float(i + 1))

        # Le buffer ne doit pas dépasser sa capacité
        assert len(buffer.buffer) == 50


class TestActionMasking:
    """Tests pour le masquage d'actions avancé."""

    def test_action_masking(self):
        """Test masquage des actions invalides."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Obtenir masque d'actions valides
        valid_mask = env._get_valid_actions_mask()

        # Vérifier que seules les actions valides sont True
        assert valid_mask[0]  # Action wait toujours valide

        # Vérifier que les actions invalides sont False
        invalid_actions = np.where(~valid_mask)[0]
        for action in invalid_actions:
            # Tenter action invalide
            _, reward, _, _, _ = env.step(action)
            assert reward == -100.0  # Pénalité pour action invalide

    def test_masked_action_selection(self):
        """Test sélection d'action avec masquage."""
        agent = ImprovedDQNAgent(state_dim=0.100, action_dim=0.100)
        state = np.random.randn(100)
        valid_actions = [0, 5, 10, 15]  # Actions valides

        # Sélectionner action avec masquage
        action = agent.select_action(state, valid_actions)

        # Vérifier que l'action est valide
        assert action in valid_actions

    def test_time_window_constraint(self):
        """Test contraintes de fenêtre temporelle."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Créer un chauffeur et un booking
        driver = {"available": True, "lat": 46.2044, "lon": 6.1432, "current_bookings": 0, "load": 0}

        booking = {"pickup_lat": 46.2044, "pickup_lon": 6.1432, "time_window_end": 30.0, "assigned": False}

        # Vérifier contrainte valide
        assert env._check_time_window_constraint(driver, booking)

        # Modifier booking pour rendre impossible
        booking["time_window_end"] = 0.0  # Fenêtre fermée
        assert not env._check_time_window_constraint(driver, booking)

    def test_driver_capacity_constraint(self):
        """Test contraintes de capacité chauffeur."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Chauffeur à capacité maximale
        driver = {
            "available": True,
            "lat": 46.2044,
            "lon": 6.1432,
            "current_bookings": 3,  # Max capacité
            "load": 0,
        }

        booking = {"pickup_lat": 46.2044, "pickup_lon": 6.1432, "time_window_end": 30.0, "assigned": False}

        # Vérifier que la contrainte est respectée
        assert not env._check_time_window_constraint(driver, booking)

    def test_get_valid_actions(self):
        """Test méthode get_valid_actions."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        valid_actions = env.get_valid_actions()

        # Vérifier que la liste contient des actions valides
        assert isinstance(valid_actions, list)
        assert len(valid_actions) > 0
        assert 0 in valid_actions  # Action wait toujours valide


class TestRewardInvariants:
    """Tests pour les invariants de récompense."""

    def test_reward_invariants(self):
        """Test invariants des récompenses."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Test invariant: Assignment toujours positif
        for _ in range(10):
            env.reset()
            # Forcer une assignation valide
            valid_actions = env._get_valid_actions_mask()
            valid_action_indices = np.where(valid_actions)[0]

            if len(valid_action_indices) > 1:  # Plus que wait
                action = valid_action_indices[1]  # Première action d'assignation
                _, reward, _, _, _ = env.step(action)

                if reward > 0:  # Si assignation réussie
                    assert reward >= 50.0  # Minimum pour assignation avec reward shaping

    def test_cancellation_penalty(self):
        """Test pénalité pour annulation."""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Laisser expirer des bookings
        for _ in range(20):  # 20 steps = 100 minutes
            _, reward, _, _, _ = env.step(0)  # Action wait

        # Vérifier que les annulations donnent des pénalités négatives
        assert reward < 0  # Pénalité pour annulation

    def test_punctuality_rewards(self):
        """Test récompenses de ponctualité."""
        reward_shaping = AdvancedRewardShaping()

        # Test ponctualité parfaite
        info_perfect = {"is_late": False, "lateness_minutes": 0, "is_outbound": True}
        reward_perfect = reward_shaping._calculate_punctuality_reward(info_perfect)
        assert reward_perfect > 0

        # Test retard ALLER (0 tolérance)
        info_late_aller = {"is_late": True, "lateness_minutes": 10, "is_outbound": True}
        reward_late_aller = reward_shaping._calculate_punctuality_reward(info_late_aller)
        assert reward_late_aller < 0

        # Test retard RETOUR (tolérance progressive)
        info_late_retour = {
            "is_late": True,
            "lateness_minutes": 10,  # Dans tolérance douce
            "is_outbound": False,
        }
        reward_late_retour = reward_shaping._calculate_punctuality_reward(info_late_retour)
        assert reward_late_retour == 0.0  # Neutre dans tolérance

    def test_distance_rewards(self):
        """Test récompenses de distance."""
        reward_shaping = AdvancedRewardShaping()

        # Test distance courte (bonus)
        info_short = {"distance_km": 2.0}
        reward_short = reward_shaping._calculate_distance_reward(info_short)
        assert reward_short > 0

        # Test distance longue (pénalité)
        info_long = {"distance_km": 20.0}
        reward_long = reward_shaping._calculate_distance_reward(info_long)
        assert reward_long < 0

    def test_equity_rewards(self):
        """Test récompenses d'équité."""
        reward_shaping = AdvancedRewardShaping()

        # Test équilibre parfait
        info_balanced = {"driver_loads": [2, 2, 2]}
        reward_balanced = reward_shaping._calculate_equity_reward(info_balanced)
        assert reward_balanced > 0

        # Test déséquilibre
        info_unbalanced = {"driver_loads": [0, 5, 10]}
        reward_unbalanced = reward_shaping._calculate_equity_reward(info_unbalanced)
        assert reward_unbalanced < 0


class TestRewardShapingConfig:
    """Tests pour la configuration du reward shaping."""

    def test_default_config(self):
        """Test configuration par défaut."""
        config = RewardShapingConfig.get_profile("DEFAULT")
        assert config["punctuality_weight"] == 1.0
        assert config["distance_weight"] == 0.5
        assert config["equity_weight"] == 0.3

    def test_punctuality_focused_config(self):
        """Test configuration focalisée ponctualité."""
        config = RewardShapingConfig.get_profile("PUNCTUALITY_FOCUSED")
        assert config["punctuality_weight"] == 1.5
        assert config["punctuality_weight"] > config["distance_weight"]

    def test_equity_focused_config(self):
        """Test configuration focalisée équité."""
        config = RewardShapingConfig.get_profile("EQUITY_FOCUSED")
        assert config["equity_weight"] == 0.6
        assert config["equity_weight"] > config["punctuality_weight"]

    def test_efficiency_focused_config(self):
        """Test configuration focalisée efficacité."""
        config = RewardShapingConfig.get_profile("EFFICIENCY_FOCUSED")
        assert config["distance_weight"] == 1.0
        assert config["distance_weight"] > config["punctuality_weight"]

    def test_invalid_profile(self):
        """Test profil invalide."""
        config = RewardShapingConfig.get_profile("INVALID_PROFILE")
        # Doit retourner la configuration par défaut
        assert config == RewardShapingConfig.DEFAULT


class TestBaselineMetrics:
    """Tests pour les métriques baseline."""

    def test_performance_baseline(self):
        """Test métriques de performance baseline."""
        agent = ImprovedDQNAgent(state_dim=0.100, action_dim=0.100)

        # Test latence d'inférence
        state = np.random.randn(100)
        import time

        start_time = time.time()
        for _ in range(100):
            agent.select_action(state, training=False)
        end_time = time.time()

        avg_latency = (end_time - start_time) / 100 * 1000  # ms
        assert avg_latency < 50.0  # Latence < 50ms

    def test_memory_usage(self):
        """Test utilisation mémoire."""
        buffer = PrioritizedReplayBuffer(10000, alpha=0.6)

        # Ajouter beaucoup de transitions
        for i in range(5000):
            state = np.random.randn(100)
            buffer.add(state, i % 100, float(i), state, False, priority=float(i + 1))

        # Vérifier que le buffer fonctionne correctement
        assert len(buffer.buffer) == 5000
        batch, _indices, _weights = buffer.sample(32)
        assert len(batch) == 32

    def test_convergence_stability(self):
        """Test stabilité de convergence."""
        agent = ImprovedDQNAgent(state_dim=0.100, action_dim=0.100)

        # Simuler plusieurs étapes d'apprentissage
        losses = []
        for _ in range(100):
            # Ajouter des transitions
            for _ in range(10):
                state = np.random.randn(100)
                action = np.random.randint(100)
                reward = float(np.random.randn())
                next_state = np.random.randn(100)
                done = np.random.choice([True, False])

                agent.store_transition(state, action, reward, next_state, done)

            # Apprendre
            if len(agent.memory) >= agent.batch_size:
                loss = agent.learn()
                losses.append(loss)

        # Vérifier que les pertes sont stables (pas d'explosion)
        if losses:
            assert all(loss < 100.0 for loss in losses)  # Pas d'explosion de gradient


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
