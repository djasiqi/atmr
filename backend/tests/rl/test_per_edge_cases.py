#!/usr/bin/env python3
"""
Tests edge cases pour le Prioritized Experience Replay (PER).

Tests spécifiques pour les cas limites identifiés par l'audit :
- PER priorities update edge cases
- Buffer overflow scenarios
- Priority calculation edge cases
- Sampling efficiency edge cases

Auteur: ATMR Project - RL Team
Date: 24 octobre 2025
"""

import numpy as np
import pytest

# Imports conditionnels
try:
    from services.rl.replay_buffer import PrioritizedReplayBuffer
except ImportError:
    PrioritizedReplayBuffer = None


class TestPERPriorityUpdateEdgeCases:
    """Tests edge cases pour la mise à jour des priorités PER."""

    @pytest.fixture
    def buffer(self):
        """Crée un buffer PER pour les tests."""
        if PrioritizedReplayBuffer is None:
            pytest.skip("PrioritizedReplayBuffer non disponible")

        return PrioritizedReplayBuffer(
            capacity=100, alpha=0.6, beta_start=0.4, beta_end=1.0
        )

    def test_priority_update_with_zero_td_error(self, buffer):
        """Test mise à jour priorité avec TD-error = 0."""
        # Ajouter une transition
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([2, 3, 4])
        done = False

        buffer.add(state, action, reward, next_state, done)

        # Mettre à jour avec TD-error = 0
        indices = [0]
        td_errors = [0.0]

        buffer.update_priorities(indices, td_errors)

        # Vérifier que la priorité est mise à jour correctement
        assert buffer.priorities[0] > 0  # Devrait être > 0 même avec TD-error = 0
        assert buffer.max_priority > 0

    def test_priority_update_with_negative_td_error(self, buffer):
        """Test mise à jour priorité avec TD-error négatif."""
        # Ajouter une transition
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([2, 3, 4])
        done = False

        buffer.add(state, action, reward, next_state, done)

        # Mettre à jour avec TD-error négatif
        indices = [0]
        td_errors = [-0.5]

        buffer.update_priorities(indices, td_errors)

        # Vérifier que la priorité est mise à jour correctement (abs)
        assert buffer.priorities[0] > 0
        assert buffer.max_priority > 0

    def test_priority_update_with_extremely_large_td_error(self, buffer):
        """Test mise à jour priorité avec TD-error très grand."""
        # Ajouter une transition
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([2, 3, 4])
        done = False

        buffer.add(state, action, reward, next_state, done)

        # Mettre à jour avec TD-error très grand
        indices = [0]
        td_errors = [1000.0]

        buffer.update_priorities(indices, td_errors)

        # Vérifier que la priorité est mise à jour correctement
        assert buffer.priorities[0] > 0
        assert buffer.max_priority > 0
        # Vérifier que la priorité n'explose pas
        assert buffer.priorities[0] < 1e6

    def test_priority_update_with_multiple_transitions(self, buffer):
        """Test mise à jour priorité avec plusieurs transitions."""
        # Ajouter plusieurs transitions
        for i in range(10):
            state = np.array([i, i + 1, i + 2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i + 1, i + 2, i + 3])
            done = i == 9

            buffer.add(state, action, reward, next_state, done)

        # Mettre à jour toutes les priorités
        indices = list(range(10))
        td_errors = [float(i) * 0.1 for i in range(10)]

        buffer.update_priorities(indices, td_errors)

        # Vérifier que toutes les priorités sont mises à jour
        for i in range(10):
            assert buffer.priorities[i] > 0

        assert buffer.max_priority > 0

    def test_priority_update_with_invalid_indices(self, buffer):
        """Test mise à jour priorité avec indices invalides."""
        # Ajouter une transition
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([2, 3, 4])
        done = False

        buffer.add(state, action, reward, next_state, done)

        # Essayer de mettre à jour avec des indices invalides
        indices = [10, 20, 30]  # Indices qui n'existent pas
        td_errors = [0.5, 0.6, 0.7]

        # Cela ne devrait pas lever d'erreur
        buffer.update_priorities(indices, td_errors)

        # Vérifier que le buffer n'est pas corrompu
        assert len(buffer.buffer) == 1
        assert len(buffer.priorities) == 1

    def test_priority_update_with_empty_indices(self, buffer):
        """Test mise à jour priorité avec liste d'indices vide."""
        # Ajouter une transition
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([2, 3, 4])
        done = False

        buffer.add(state, action, reward, next_state, done)

        # Essayer de mettre à jour avec des indices vides
        indices = []
        td_errors = []

        # Cela ne devrait pas lever d'erreur
        buffer.update_priorities(indices, td_errors)

        # Vérifier que le buffer n'est pas corrompu
        assert len(buffer.buffer) == 1
        assert len(buffer.priorities) == 1

    def test_priority_update_with_nan_td_error(self, buffer):
        """Test mise à jour priorité avec TD-error NaN."""
        # Ajouter une transition
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([2, 3, 4])
        done = False

        buffer.add(state, action, reward, next_state, done)

        # Mettre à jour avec TD-error NaN
        indices = [0]
        td_errors = [np.nan]

        # Cela devrait gérer NaN correctement
        buffer.update_priorities(indices, td_errors)

        # Vérifier que la priorité est mise à jour correctement
        assert buffer.priorities[0] > 0
        assert not np.isnan(buffer.priorities[0])

    def test_priority_update_with_inf_td_error(self, buffer):
        """Test mise à jour priorité avec TD-error infini."""
        # Ajouter une transition
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([2, 3, 4])
        done = False

        buffer.add(state, action, reward, next_state, done)

        # Mettre à jour avec TD-error infini
        indices = [0]
        td_errors = [np.inf]

        # Cela devrait gérer infini correctement
        buffer.update_priorities(indices, td_errors)

        # Vérifier que la priorité est mise à jour correctement
        assert buffer.priorities[0] > 0
        assert not np.isinf(buffer.priorities[0])

    def test_priority_update_consistency(self, buffer):
        """Test cohérence des mises à jour de priorité."""
        # Ajouter plusieurs transitions
        for i in range(5):
            state = np.array([i, i + 1, i + 2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i + 1, i + 2, i + 3])
            done = i == 4

            buffer.add(state, action, reward, next_state, done)

        # Première mise à jour
        indices = [0, 1, 2]
        td_errors = [0.5, 0.6, 0.7]
        buffer.update_priorities(indices, td_errors)

        priorities_1 = buffer.priorities.copy()
        max_priority_1 = buffer.max_priority

        # Deuxième mise à jour
        indices = [1, 2, 3]
        td_errors = [0.8, 0.9, 1.0]
        buffer.update_priorities(indices, td_errors)

        priorities_2 = buffer.priorities.copy()
        max_priority_2 = buffer.max_priority

        # Vérifier que les priorités ont été mises à jour
        assert not np.array_equal(priorities_1, priorities_2)
        assert max_priority_2 >= max_priority_1

    def test_priority_update_with_mixed_td_errors(self, buffer):
        """Test mise à jour priorité avec TD-errors mixtes."""
        # Ajouter plusieurs transitions
        for i in range(5):
            state = np.array([i, i + 1, i + 2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i + 1, i + 2, i + 3])
            done = i == 4

            buffer.add(state, action, reward, next_state, done)

        # Mettre à jour avec TD-errors mixtes (positifs, négatifs, zéro)
        indices = [0, 1, 2, 3, 4]
        td_errors = [0.5, -0.3, 0.0, 1.2, -0.8]

        buffer.update_priorities(indices, td_errors)

        # Vérifier que toutes les priorités sont positives
        for i in range(5):
            assert buffer.priorities[i] > 0
            assert not np.isnan(buffer.priorities[i])
            assert not np.isinf(buffer.priorities[i])

        assert buffer.max_priority > 0


class TestPERBufferOverflowEdgeCases:
    """Tests edge cases pour le débordement du buffer PER."""

    @pytest.fixture
    def small_buffer(self):
        """Crée un petit buffer PER pour tester le débordement."""
        if PrioritizedReplayBuffer is None:
            pytest.skip("PrioritizedReplayBuffer non disponible")

        return PrioritizedReplayBuffer(
            capacity=5,  # Petit buffer pour tester le débordement
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
        )

    def test_buffer_overflow_priority_update(self, small_buffer):
        """Test mise à jour priorité après débordement du buffer."""
        # Remplir le buffer
        for i in range(7):  # Plus que la capacité
            state = np.array([i, i + 1, i + 2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i + 1, i + 2, i + 3])
            done = i == 6

            small_buffer.add(state, action, reward, next_state, done)

        # Vérifier que le buffer a la bonne taille
        assert len(small_buffer.buffer) == 5
        assert len(small_buffer.priorities) == 5

        # Mettre à jour toutes les priorités
        indices = list(range(5))
        td_errors = [float(i) * 0.1 for i in range(5)]

        small_buffer.update_priorities(indices, td_errors)

        # Vérifier que toutes les priorités sont mises à jour
        for i in range(5):
            assert small_buffer.priorities[i] > 0

        assert small_buffer.max_priority > 0

    def test_buffer_overflow_sampling(self, small_buffer):
        """Test échantillonnage après débordement du buffer."""
        # Remplir le buffer
        for i in range(7):  # Plus que la capacité
            state = np.array([i, i + 1, i + 2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i + 1, i + 2, i + 3])
            done = i == 6

            small_buffer.add(state, action, reward, next_state, done)

        # Échantillonner
        batch_size = 3
        batch, weights, indices = small_buffer.sample(batch_size)

        # Vérifier que l'échantillonnage fonctionne
        assert len(batch) == batch_size
        assert len(weights) == batch_size
        assert len(indices) == batch_size

        # Vérifier que tous les indices sont valides
        for idx in indices:
            assert 0 <= int(idx) < 5

    def test_buffer_overflow_max_priority(self, small_buffer):
        """Test max_priority après débordement du buffer."""
        # Remplir le buffer
        for i in range(7):  # Plus que la capacité
            state = np.array([i, i + 1, i + 2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i + 1, i + 2, i + 3])
            done = i == 6

            small_buffer.add(state, action, reward, next_state, done)

        # Vérifier que max_priority est correct
        assert small_buffer.max_priority > 0
        assert small_buffer.max_priority == max(small_buffer.priorities)


class TestPERSamplingEdgeCases:
    """Tests edge cases pour l'échantillonnage PER."""

    @pytest.fixture
    def buffer(self):
        """Crée un buffer PER pour les tests."""
        if PrioritizedReplayBuffer is None:
            pytest.skip("PrioritizedReplayBuffer non disponible")

        return PrioritizedReplayBuffer(
            capacity=100, alpha=0.6, beta_start=0.4, beta_end=1.0
        )

    def test_sampling_with_zero_priorities(self, buffer):
        """Test échantillonnage avec priorités zéro."""
        # Ajouter une transition
        state = np.array([1, 2, 3])
        action = 1
        reward = 0.5
        next_state = np.array([2, 3, 4])
        done = False

        buffer.add(state, action, reward, next_state, done)

        # Mettre la priorité à zéro
        buffer.priorities[0] = 0.0

        # Échantillonner
        batch_size = 1
        batch, weights, indices = buffer.sample(batch_size)

        # Vérifier que l'échantillonnage fonctionne quand même
        assert len(batch) == batch_size
        assert len(weights) == batch_size
        assert len(indices) == batch_size

    def test_sampling_with_identical_priorities(self, buffer):
        """Test échantillonnage avec priorités identiques."""
        # Ajouter plusieurs transitions avec priorités identiques
        for i in range(5):
            state = np.array([i, i + 1, i + 2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i + 1, i + 2, i + 3])
            done = i == 4

            buffer.add(state, action, reward, next_state, done)

        # Mettre toutes les priorités identiques
        for i in range(5):
            buffer.priorities[i] = 1.0

        # Échantillonner plusieurs fois
        for _ in range(10):
            batch_size = 3
            batch, weights, indices = buffer.sample(batch_size)

            # Vérifier que l'échantillonnage fonctionne
            assert len(batch) == batch_size
            assert len(weights) == batch_size
            assert len(indices) == batch_size

    def test_sampling_with_extreme_priorities(self, buffer):
        """Test échantillonnage avec priorités extrêmes."""
        # Ajouter plusieurs transitions
        for i in range(5):
            state = np.array([i, i + 1, i + 2])
            action = i % 3
            reward = i * 0.1
            next_state = np.array([i + 1, i + 2, i + 3])
            done = i == 4

            buffer.add(state, action, reward, next_state, done)

        # Mettre des priorités extrêmes
        buffer.priorities[0] = 1e-6  # Très petite
        buffer.priorities[1] = 1e6  # Très grande
        buffer.priorities[2] = 1.0  # Normale
        buffer.priorities[3] = 0.5  # Normale
        buffer.priorities[4] = 2.0  # Normale

        # Mettre à jour l'arbre après modification des priorités
        for i in range(5):
            buffer._update_tree(i, buffer.priorities[i])
            buffer.max_priority = max(buffer.max_priority, buffer.priorities[i])

        # Échantillonner plusieurs fois
        for _ in range(10):
            batch_size = 3
            batch, weights, indices = buffer.sample(batch_size)

            # Vérifier que l'échantillonnage fonctionne
            assert len(batch) == batch_size
            assert len(weights) == batch_size
            assert len(indices) == batch_size

            # Vérifier que les poids sont valides
            for weight in weights:
                assert weight > 0
                assert not np.isnan(weight)
                assert not np.isinf(weight)
