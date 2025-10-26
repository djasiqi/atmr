#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
Tests pour le buffer N-step et l'intégration dans l'agent DQN.

Tests complets pour valider le fonctionnement du N-step learning
et son impact sur l'efficacité d'échantillonnage.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
import unittest
from unittest.mock import Mock, patch

import numpy as np

# pyright: reportMissingImports=false
try:
    import torch
except ImportError:
    torch = None

from services.rl.improved_dqn_agent import ImprovedDQNAgent
from services.rl.n_step_buffer import NStepBuffer, NStepPrioritizedBuffer, create_n_step_buffer


class TestNStepBuffer(unittest.TestCase):
    """Tests pour le buffer N-step standard."""
    
    def setUp(self):
        """Configuration des tests."""
        self.buffer = NStepBuffer(capacity=0.1000, n_step=3, gamma=0.99)
        self.logger = logging.getLogger(__name__)
    
    def test_buffer_initialization(self):
        """Test l'initialisation du buffer."""
        assert self.buffer.capacity == 1000
        assert self.buffer.n_step == 3
        assert self.buffer.gamma == 0.99
        assert len(self.buffer) == 0
        assert len(self.buffer.temp_buffer) == 0
    
    def test_add_single_transition(self):
        """Test l'ajout d'une seule transition."""
        state = np.random.randn(10)
        action = 1
        reward = 0.5
        next_state = np.random.randn(10)
        done = False
        
        self.buffer.add_transition(state, action, reward, next_state, done)
        
        # Le buffer temporaire devrait contenir la transition
        assert len(self.buffer.temp_buffer) == 1
        # Le buffer principal devrait être vide (pas encore n_step transitions)
        assert len(self.buffer) == 0
    
    def test_n_step_calculation(self):
        """Test le calcul des retours N-step."""
        # Ajouter exactement n_step transitions
        for i in range(3):
            state = np.random.randn(10)
            action = i
            reward = 1.0  # Récompense constante
            next_state = np.random.randn(10)
            done = (i == 2)  # Terminer à la dernière transition
            
            self.buffer.add_transition(state, action, reward, next_state, done)
        
        # Le buffer principal devrait maintenant contenir les transitions N-step
        assert len(self.buffer) == 3
        assert len(self.buffer.temp_buffer) == 0
        
        # Vérifier que les retours N-step sont calculés
        batch, _weights = self.buffer.sample(3)
        assert len(batch) == 3
        
        for transition in batch:
            assert "n_step_return" in transition
            assert "n_step" in transition
            assert "state" in transition
            assert "action" in transition
            assert "next_state" in transition
    
    def test_n_step_return_calculation(self):
        """Test le calcul précis des retours N-step."""
        buffer = NStepBuffer(capacity=0.100, n_step=2, gamma=0.9)
        
        # Ajouter 2 transitions avec des récompenses connues
        buffer.add_transition(
            np.array([1.0]), 0, 1.0, np.array([2.0]), False
        )
        buffer.add_transition(
            np.array([2.0]), 1, 2.0, np.array([3.0]), True
        )
        
        # Le retour N-step pour la première transition devrait être:
        # 1.0 + 0.9 * 2.0 = 2.8
        batch, _ = buffer.sample(2)
        
        first_transition = batch[0]
        expected_return = 1.0 + 0.9 * 2.0
        self.assertAlmostEqual(first_transition["n_step_return"], expected_return, places=5)
    
    def test_episode_termination(self):
        """Test le traitement des épisodes terminés."""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)
        
        # Ajouter 2 transitions puis terminer l'épisode
        buffer.add_transition(np.array([1.0]), 0, 1.0, np.array([2.0]), False)
        buffer.add_transition(np.array([2.0]), 1, 2.0, np.array([3.0]), True)
        
        # Le buffer devrait traiter les transitions même si < n_step
        assert len(buffer) == 2
        assert len(buffer.temp_buffer) == 0
    
    def test_sample_empty_buffer(self):
        """Test l'échantillonnage d'un buffer vide."""
        batch, weights = self.buffer.sample(10)
        assert len(batch) == 0
        assert len(weights) == 0
    
    def test_sample_partial_buffer(self):
        """Test l'échantillonnage d'un buffer partiellement rempli."""
        # Ajouter quelques transitions
        for i in range(5):
            state = np.random.randn(10)
            action = i
            reward = 1.0
            next_state = np.random.randn(10)
            done = (i == 4)
            
            self.buffer.add_transition(state, action, reward, next_state, done)
        
        # Échantillonner un batch plus petit que le buffer
        batch, weights = self.buffer.sample(3)
        assert len(batch) == 3
        assert len(weights) == 3
        
        # Vérifier que tous les poids sont 1.0 (uniforme)
        for weight in weights:
            assert weight == 1.0
    
    def test_statistics(self):
        """Test les statistiques du buffer."""
        # Ajouter quelques transitions
        for i in range(5):
            state = np.random.randn(10)
            action = i
            reward = 1.0
            next_state = np.random.randn(10)
            done = (i == 4)
            
            self.buffer.add_transition(state, action, reward, next_state, done)
        
        stats = self.buffer.get_statistics()
        
        assert "buffer_size" in stats
        assert "temp_buffer_size" in stats
        assert "total_added" in stats
        assert "total_completed" in stats
        assert "completion_rate" in stats
        
        assert stats["total_added"] == 5
        assert stats["completion_rate"] == stats["total_completed"] / stats["total_added"]
    
    def test_clear_buffer(self):
        """Test le vidage du buffer."""
        # Ajouter quelques transitions
        for i in range(3):
            state = np.random.randn(10)
            action = i
            reward = 1.0
            next_state = np.random.randn(10)
            done = (i == 2)
            
            self.buffer.add_transition(state, action, reward, next_state, done)
        
        # Vider le buffer
        self.buffer.clear()
        
        assert len(self.buffer) == 0
        assert len(self.buffer.temp_buffer) == 0
        assert self.buffer.total_added == 0
        assert self.buffer.total_completed == 0


class TestNStepPrioritizedBuffer(unittest.TestCase):
    """Tests pour le buffer N-step priorisé."""
    
    def setUp(self):
        """Configuration des tests."""
        self.buffer = NStepPrioritizedBuffer(
            capacity=0.1000, n_step=3, gamma=0.99,
            alpha=0.6, beta_start=0.4, beta_end=1.0
        )
    
    def test_prioritized_initialization(self):
        """Test l'initialisation du buffer priorisé."""
        assert self.buffer.alpha == 0.6
        assert self.buffer.beta_start == 0.4
        assert self.buffer.beta_end == 1.0
        assert self.buffer.beta == 0.4
        assert self.buffer.max_priority == 1.0
    
    def test_add_transition_with_td_error(self):
        """Test l'ajout de transition avec erreur TD."""
        state = np.random.randn(10)
        action = 1
        reward = 0.5
        next_state = np.random.randn(10)
        done = False
        td_error = 0.8
        
        self.buffer.add_transition(state, action, reward, next_state, done, None, td_error)
        
        # Vérifier que la priorité est mise à jour
        assert self.buffer.max_priority > 1.0
    
    def test_prioritized_sampling(self):
        """Test l'échantillonnage priorisé."""
        # Ajouter plusieurs transitions avec différentes priorités
        for i in range(5):
            state = np.random.randn(10)
            action = i
            reward = 1.0
            next_state = np.random.randn(10)
            done = (i == 4)
            td_error = 0.5 + i * 0.1  # Priorités croissantes
            
            self.buffer.add_transition(state, action, reward, next_state, done, None, td_error)
        
        # Échantillonner plusieurs fois et vérifier la distribution
        batch, weights = self.buffer.sample(3)
        
        assert len(batch) == 3
        assert len(weights) == 3
        
        # Les poids devraient être normalisés
        max_weight = max(weights)
        for weight in weights:
            assert weight <= max_weight
            assert weight >= 0.0
    
    def test_update_priorities(self):
        """Test la mise à jour des priorités."""
        # Ajouter quelques transitions
        for i in range(3):
            state = np.random.randn(10)
            action = i
            reward = 1.0
            next_state = np.random.randn(10)
            done = (i == 2)
            
            self.buffer.add_transition(state, action, reward, next_state, done)
        
        # Mettre à jour les priorités
        indices = [0, 1, 2]
        td_errors = [0.5, 1.0, 1.5]
        
        self.buffer.update_priorities(indices, td_errors)
        
        # Vérifier que les priorités sont mises à jour
        assert self.buffer.max_priority > 1.0


class TestNStepIntegration(unittest.TestCase):
    """Tests d'intégration avec l'agent DQN."""
    
    @unittest.skipIf(torch is None, "PyTorch not available")
    def test_agent_with_n_step(self):
        """Test l'agent DQN avec N-step learning."""
        # Créer un agent avec N-step
        agent = ImprovedDQNAgent(
            state_dim=10,
            action_dim=5,
            use_n_step=True,
            n_step=3,
            n_step_gamma=0.99,
            use_prioritized_replay=True
        )
        
        # Vérifier que le buffer est de type N-step
        assert isinstance(agent.memory, NStepPrioritizedBuffer)
        assert agent.use_n_step
        assert agent.n_step == 3
    
    @unittest.skipIf(torch is None, "PyTorch not available")
    def test_agent_without_n_step(self):
        """Test l'agent DQN sans N-step (mode standard)."""
        # Créer un agent sans N-step
        agent = ImprovedDQNAgent(
            state_dim=10,
            action_dim=5,
            use_n_step=False,
            use_prioritized_replay=True
        )
        
        # Vérifier que le buffer est de type PER standard
        from services.rl.replay_buffer import PrioritizedReplayBuffer
        assert isinstance(agent.memory, PrioritizedReplayBuffer)
        assert not agent.use_n_step
    
    @unittest.skipIf(torch is None, "PyTorch not available")
    def test_store_transition_n_step(self):
        """Test le stockage de transitions avec N-step."""
        agent = ImprovedDQNAgent(
            state_dim=10,
            action_dim=5,
            use_n_step=True,
            n_step=3
        )
        
        # Stocker quelques transitions
        for i in range(5):
            state = np.random.randn(10)
            action = i % 5
            reward = 1.0
            next_state = np.random.randn(10)
            done = (i == 4)
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Vérifier que les transitions sont stockées
        assert len(agent.memory) > 0
    
    @unittest.skipIf(torch is None, "PyTorch not available")
    def test_learn_with_n_step(self):
        """Test l'apprentissage avec N-step."""
        agent = ImprovedDQNAgent(
            state_dim=10,
            action_dim=5,
            use_n_step=True,
            n_step=3,
            batch_size=32
        )
        
        # Remplir le buffer avec suffisamment de transitions
        for i in range(100):
            state = np.random.randn(10)
            action = i % 5
            reward = np.random.randn()
            next_state = np.random.randn(10)
            done = (i % 20 == 19)  # Terminer tous les 20 steps
            
            agent.store_transition(state, action, reward, next_state, done)
        
        # Effectuer un apprentissage
        loss = agent.learn()
        
        # Vérifier que l'apprentissage s'est bien déroulé
        assert isinstance(loss, float)
        assert loss >= 0.0


class TestNStepPerformance(unittest.TestCase):
    """Tests de performance pour le N-step learning."""
    
    def test_sample_efficiency_comparison(self):
        """Test la comparaison d'efficacité d'échantillonnage."""
        # Buffer standard
        standard_buffer = NStepBuffer(capacity=0.1000, n_step=1, gamma=0.99)
        
        # Buffer N-step
        n_step_buffer = NStepBuffer(capacity=0.1000, n_step=3, gamma=0.99)
        
        # Ajouter les mêmes transitions aux deux buffers
        for i in range(10):
            state = np.random.randn(10)
            action = i % 5
            reward = 1.0
            next_state = np.random.randn(10)
            done = (i == 9)
            
            standard_buffer.add_transition(state, action, reward, next_state, done)
            n_step_buffer.add_transition(state, action, reward, next_state, done)
        
        # Comparer les statistiques
        standard_stats = standard_buffer.get_statistics()
        n_step_stats = n_step_buffer.get_statistics()
        
        # Le buffer N-step devrait avoir un taux de completion différent
        assert standard_stats["completion_rate"] != n_step_stats["completion_rate"]
    
    def test_memory_usage(self):
        """Test l'utilisation mémoire des buffers."""
        import gc
        
        # Créer un buffer et mesurer la mémoire
        buffer = NStepBuffer(capacity=0.10000, n_step=3, gamma=0.99)
        
        # Ajouter beaucoup de transitions
        for i in range(1000):
            state = np.random.randn(50)  # États plus grands
            action = i % 10
            reward = np.random.randn()
            next_state = np.random.randn(50)
            done = (i % 100 == 99)
            
            buffer.add_transition(state, action, reward, next_state, done)
        
        # Vérifier que le buffer fonctionne toujours
        stats = buffer.get_statistics()
        assert stats["total_added"] > 0
        assert stats["total_completed"] > 0
        
        # Nettoyer
        buffer.clear()
        gc.collect()


def run_n_step_tests():
    """Lance tous les tests N-step."""
    logging.basicConfig(level=logging.INFO)
    
    # Créer une suite de tests
    test_suite = unittest.TestSuite()
    
    # Ajouter les tests
    test_classes = [
        TestNStepBuffer,
        TestNStepPrioritizedBuffer,
        TestNStepIntegration,
        TestNStepPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Exécuter les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_n_step_tests()
    if success:
        print("✅ Tous les tests N-step ont réussi !")
    else:
        print("❌ Certains tests N-step ont échoué.")
