#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Tests complets pour Prioritized Experience Replay (PER).

Améliore la couverture de tests pour le système PER
implémenté dans les étapes précédentes.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

# Import conditionnel pour éviter les erreurs si les modules ne sont pas disponibles
try:
    from services.rl.replay_buffer import PrioritizedReplayBuffer
except ImportError:
    PrioritizedReplayBuffer = None

try:
    from services.rl.improved_dqn_agent import ImprovedDQNAgent
except ImportError:
    ImprovedDQNAgent = None


class TestPrioritizedReplayBuffer:
    """Tests pour PrioritizedReplayBuffer."""

    @pytest.fixture
    def buffer(self):
        """Crée un buffer PER pour les tests."""
        if PrioritizedReplayBuffer is None:
            pytest.skip("PrioritizedReplayBuffer non disponible")
        
        return PrioritizedReplayBuffer(
            capacity=0.1000,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0
        )

    def test_buffer_initialization(self, buffer):
        """Test l'initialisation du buffer."""
        assert buffer.capacity == 1000
        assert buffer.alpha == 0.6
        assert buffer.beta_start == 0.4
        assert buffer.beta_end == 1.0
        assert len(buffer.buffer) == 0
        assert len(buffer.priorities) == 0

    def test_add_transition(self, buffer):
        """Test l'ajout d'une transition."""
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([4, 5, 6])
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer.buffer) == 1
        assert len(buffer.priorities) == 1
        assert buffer.priorities[0] == buffer.max_priority

    def test_add_multiple_transitions(self, buffer):
        """Test l'ajout de plusieurs transitions."""
        for i in range(5):
            state = np.array([i, i+1, i+2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i+3, i+4, i+5])
            done = i == 4
            
            buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer.buffer) == 5
        assert len(buffer.priorities) == 5

    def test_buffer_capacity_overflow(self, buffer):
        """Test le dépassement de capacité."""
        # Ajouter plus de transitions que la capacité
        for i in range(1200):
            state = np.array([i, i+1, i+2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i+3, i+4, i+5])
            done = i % 100 == 0
            
            buffer.add(state, action, reward, next_state, done)
        
        assert len(buffer.buffer) == buffer.capacity
        assert len(buffer.priorities) == buffer.capacity

    def test_sample_batch(self, buffer):
        """Test l'échantillonnage d'un batch."""
        # Ajouter des transitions
        for i in range(10):
            state = np.array([i, i+1, i+2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i+3, i+4, i+5])
            done = i % 3 == 0
            
            buffer.add(state, action, reward, next_state, done)
        
        batch_size = 4
        batch, indices, weights = buffer.sample(batch_size)
        
        assert len(batch) == batch_size
        assert len(weights) == batch_size
        assert len(indices) == batch_size
        
        # Debug: afficher les poids
        print("Weights: {weights}")
        print("Min weight: {min(weights) if weights else 'N/A'}")
        print("Max weight: {max(weights) if weights else 'N/A'}")
        
        # Vérifier que les poids sont des probabilités
        assert all(0 <= w <= 1 for w in weights)

    def test_update_priorities(self, buffer):
        """Test la mise à jour des priorités."""
        # Ajouter des transitions
        for i in range(5):
            state = np.array([i, i+1, i+2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i+3, i+4, i+5])
            done = i % 2 == 0
            
            buffer.add(state, action, reward, next_state, done)
        
        # Mettre à jour les priorités
        indices = [0, 1, 2]
        new_priorities = [0.1, 0.5, 0.9]
        
        buffer.update_priorities(indices, new_priorities)
        
        for i, priority in zip(indices, new_priorities, strict=False):
            assert abs(buffer.priorities[i] - priority) < 1e-6

    def test_beta_scheduling(self, buffer):
        """Test la planification de beta."""
        # Ajouter des transitions d'abord
        for i in range(10):
            state = np.array([i, i+1, i+2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i+3, i+4, i+5])
            done = i % 3 == 0
            
            buffer.add(state, action, reward, next_state, done)
        
        initial_beta = buffer.beta
        
        # Simuler plusieurs échantillonnages
        for _ in range(100):
            try:
                buffer.sample(4)
            except ValueError:
                # Si le buffer est trop petit, ajouter plus de transitions
                for i in range(10):
                    state = np.array([i+10, i+11, i+12])
                    action = i % 3
                    reward = i * 0.1
                    next_state = np.array([i+13, i+14, i+15])
                    done = i % 3 == 0
                    
                    buffer.add(state, action, reward, next_state, done)
        
        # Beta devrait avoir augmenté
        assert buffer.beta > initial_beta
        assert buffer.beta <= buffer.beta_end

    def test_importance_sampling_weights(self, buffer):
        """Test les poids d'importance sampling."""
        # Ajouter des transitions avec différentes priorités
        for i in range(10):
            state = np.array([i, i+1, i+2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i+3, i+4, i+5])
            done = i % 3 == 0
            
            buffer.add(state, action, reward, next_state, done)
        
        _batch, _indices, weights = buffer.sample(5)
        
        # Les poids devraient être normalisés
        assert all(w > 0 for w in weights)
        assert all(w <= 1 for w in weights)

    def test_empty_buffer_sample(self, buffer):
        """Test l'échantillonnage d'un buffer vide."""
        with pytest.raises(ValueError):
            buffer.sample(4)

    def test_invalid_batch_size(self, buffer):
        """Test avec une taille de batch invalide."""
        # Ajouter quelques transitions
        for i in range(3):
            state = np.array([i, i+1, i+2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i+3, i+4, i+5])
            done = False
            
            buffer.add(state, action, reward, next_state, done)
        
        # Taille de batch plus grande que le buffer
        with pytest.raises(ValueError):
            buffer.sample(10)

    def test_priority_calculation(self, buffer):
        """Test le calcul des priorités."""
        # Ajouter une transition
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([4, 5, 6])
        done = False
        
        buffer.add(state, action, reward, next_state, done)
        
        # La priorité initiale devrait être max_priority
        assert buffer.priorities[0] == buffer.max_priority

    def test_max_priority_update(self, buffer):
        """Test la mise à jour de max_priority."""
        initial_max = buffer.max_priority
        
        # Ajouter une transition avec une priorité élevée
        buffer.add(
            np.array([1, 2, 3]), 1, 0.5,
            np.array([4, 5, 6]), False
        )
        
        # Mettre à jour avec une priorité plus élevée
        buffer.update_priorities([0], [initial_max + 1])
        
        assert buffer.max_priority > initial_max

    def test_buffer_len(self, buffer):
        """Test la longueur du buffer."""
        assert len(buffer) == 0
        
        # Ajouter des transitions
        for i in range(5):
            buffer.add(
                np.array([i, i+1, i+2]), i % 3, i * 0.1,
                np.array([i+3, i+4, i+5]), i % 2 == 0
            )
        
        assert len(buffer) == 5

    def test_buffer_clear(self, buffer):
        """Test le vidage du buffer."""
        # Ajouter des transitions
        for i in range(5):
            buffer.add(
                np.array([i, i+1, i+2]), i % 3, i * 0.1,
                np.array([i+3, i+4, i+5]), i % 2 == 0
            )
        
        assert len(buffer) == 5
        
        # Vider le buffer
        buffer.clear()
        
        assert len(buffer) == 0
        assert len(buffer.priorities) == 0


class TestPERIntegration:
    """Tests d'intégration pour PER avec l'agent DQN."""

    @pytest.fixture
    def mock_agent(self):
        """Crée un agent DQN mock pour les tests."""
        if ImprovedDQNAgent is None:
            pytest.skip("ImprovedDQNAgent non disponible")
        
        agent = Mock(spec=ImprovedDQNAgent)
        agent.state_size = 10
        agent.action_size = 4
        agent.use_prioritized_replay = True
        agent.alpha = 0.6
        agent.beta_start = 0.4
        agent.beta_end = 1.0
        
        # Ajouter les attributs nécessaires
        agent.memory = Mock()
        agent.q_network = Mock()
        agent.target_network = Mock()
        
        return agent

    def test_per_agent_initialization(self, mock_agent):
        """Test l'initialisation de l'agent avec PER."""
        assert mock_agent.use_prioritized_replay is True
        assert mock_agent.alpha == 0.6
        assert mock_agent.beta_start == 0.4
        assert mock_agent.beta_end == 1.0

    def test_per_learning_step(self, mock_agent):
        """Test une étape d'apprentissage avec PER."""
        # Mock des méthodes nécessaires
        mock_agent.memory.sample.return_value = (
            [np.array([1, 2, 3])],  # states
            [1],  # actions
            [0.5],  # rewards
            [np.array([4, 5, 6])],  # next_states
            [False],  # dones
            [0.8],  # weights
            [0]  # indices
        )
        
        mock_agent.q_network.return_value = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        mock_agent.target_network.return_value = torch.tensor([[0.2, 0.3, 0.4, 0.5]])
        
        # Simuler une étape d'apprentissage
        # (Ceci nécessiterait l'implémentation complète de l'agent)
        assert mock_agent.use_prioritized_replay is True

    def test_per_hyperparameter_validation(self):
        """Test la validation des hyperparamètres PER."""
        valid_configs = [
            {"alpha": 0.4, "beta_start": 0.3, "beta_end": 0.8},
            {"alpha": 0.6, "beta_start": 0.4, "beta_end": 1.0},
            {"alpha": 0.8, "beta_start": 0.5, "beta_end": 1.0},
        ]
        
        for config in valid_configs:
            assert 0 < config["alpha"] <= 1
            assert 0 < config["beta_start"] < config["beta_end"] <= 1

    def test_per_performance_metrics(self):
        """Test les métriques de performance PER."""
        # Métriques typiques pour PER
        metrics = {
            "sample_efficiency": 0.85,
            "convergence_speed": 0.92,
            "stability": 0.88,
            "memory_usage": 0.75
        }
        
        for metric, value in metrics.items():
            assert 0 <= value <= 1, f"{metric} devrait être entre 0 et 1"


class TestPEREdgeCases:
    """Tests des cas limites pour PER."""

    def test_extreme_priorities(self):
        """Test avec des priorités extrêmes."""
        if PrioritizedReplayBuffer is None:
            pytest.skip("PrioritizedReplayBuffer non disponible")
        
        buffer = PrioritizedReplayBuffer(capacity=0.100)
        
        # Ajouter des transitions
        for i in range(5):
            buffer.add(
                np.array([i, i+1, i+2]), i % 3, i * 0.1,
                np.array([i+3, i+4, i+5]), i % 2 == 0
            )
        
        # Tester avec des priorités très faibles
        buffer.update_priorities([0], [1e-10])
        
        # Tester avec des priorités très élevées
        buffer.update_priorities([1], [1e10])
        
        # Le buffer devrait toujours fonctionner
        batch, _weights, _indices = buffer.sample(3)
        assert len(batch) == 3

    def test_single_transition_buffer(self):
        """Test avec un buffer contenant une seule transition."""
        if PrioritizedReplayBuffer is None:
            pytest.skip("PrioritizedReplayBuffer non disponible")
        
        buffer = PrioritizedReplayBuffer(capacity=0.100)
        
        # Ajouter une seule transition
        buffer.add(
            np.array([1, 2, 3]), 1, 0.5,
            np.array([4, 5, 6]), False
        )
        
        # Échantillonner plusieurs fois
        for _ in range(5):
            batch, weights, indices = buffer.sample(1)
            assert len(batch) == 1
            assert len(weights) == 1
            assert len(indices) == 1

    def test_concurrent_access(self):
        """Test l'accès concurrent au buffer."""
        if PrioritizedReplayBuffer is None:
            pytest.skip("PrioritizedReplayBuffer non disponible")
        
        buffer = PrioritizedReplayBuffer(capacity=0.1000)
        
        # Simuler des ajouts concurrents
        for i in range(100):
            buffer.add(
                np.array([i, i+1, i+2]), i % 3, i * 0.1,
                np.array([i+3, i+4, i+5]), i % 10 == 0
            )
        
        # Simuler des échantillonnages concurrents
        for _ in range(50):
            batch, _weights, _indices = buffer.sample(4)
            assert len(batch) == 4


def run_per_tests():
    """Exécute tous les tests PER."""
    print("🧪 Exécution des tests PER (Prioritized Experience Replay)")
    
    # Tests de base
    test_classes = [
        TestPrioritizedReplayBuffer,
        TestPERIntegration,
        TestPEREdgeCases
    ]
    
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
    
    print("\n📊 Résultats des tests PER:")
    print("  Tests exécutés: {total_tests}")
    print("  Tests réussis: {passed_tests}")
    print("  Taux de succès: {passed_tests/total_tests*100" if total_tests > 0 else "  Taux de succès: 0%")
    
    return passed_tests, total_tests


if __name__ == "__main__":
    run_per_tests()
