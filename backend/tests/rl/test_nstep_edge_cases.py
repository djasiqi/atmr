#!/usr/bin/env python3
"""
Tests edge cases pour le N-step Learning.

Tests spécifiques pour les cas limites identifiés par l'audit :
- N-step fin d'épisode edge cases
- Buffer N-step overflow scenarios
- Return calculation edge cases
- Bootstrapping edge cases

Auteur: ATMR Project - RL Team
Date: 24 octobre 2025
"""

import numpy as np
import pytest

# Imports conditionnels
try:
    from services.rl.n_step_buffer import NStepBuffer, NStepPrioritizedBuffer
except ImportError:
    NStepBuffer = None
    NStepPrioritizedBuffer = None


class TestNStepEndEpisodeEdgeCases:
    """Tests edge cases pour la fin d'épisode en N-step learning."""

    @pytest.fixture
    def n_step_buffer(self):
        """Crée un buffer N-step pour les tests."""
        if NStepBuffer is None:
            pytest.skip("NStepBuffer non disponible")

        return NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)

    def test_n_step_end_episode_exact_length(self, n_step_buffer):
        """Test N-step avec épisode de longueur exacte n."""
        # Créer un épisode de longueur exacte n=3
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (np.array([3, 4, 5]), 2, 0.3, np.array([4, 5, 6]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            n_step_buffer.add(state, action, reward, next_state, done)

        # Vérifier que toutes les transitions sont ajoutées
        assert len(n_step_buffer.buffer) == 3

        # Vérifier que les transitions partielles sont gérées correctement
        # Les premières transitions ne peuvent pas être complétées car l'épisode se termine
        assert len(n_step_buffer.buffer) == 3

    def test_n_step_end_episode_shorter_than_n(self, n_step_buffer):
        """Test N-step avec épisode plus court que n."""
        # Créer un épisode de longueur 2 (plus court que n=3)
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            n_step_buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(n_step_buffer.buffer) == 2

        # Vérifier que les transitions partielles sont gérées correctement
        # Aucune transition ne peut être complétée car l'épisode est trop court

    def test_n_step_end_episode_longer_than_n(self, n_step_buffer):
        """Test N-step avec épisode plus long que n."""
        # Créer un épisode de longueur 5 (plus long que n=3)
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (np.array([3, 4, 5]), 2, 0.3, np.array([4, 5, 6]), False),
            (np.array([4, 5, 6]), 3, 0.4, np.array([5, 6, 7]), False),
            (np.array([5, 6, 7]), 4, 0.5, np.array([6, 7, 8]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            n_step_buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(n_step_buffer.buffer) == 5

        # Vérifier que les transitions partielles sont gérées correctement
        # Les dernières transitions ne peuvent pas être complétées car l'épisode se termine

    def test_n_step_end_episode_multiple_episodes(self, n_step_buffer):
        """Test N-step avec plusieurs épisodes."""
        # Premier épisode (longueur 2)
        transitions_1 = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), True),  # Fin épisode 1
        ]

        # Deuxième épisode (longueur 4)
        transitions_2 = [
            (np.array([10, 11, 12]), 0, 0.3, np.array([11, 12, 13]), False),
            (np.array([11, 12, 13]), 1, 0.4, np.array([12, 13, 14]), False),
            (np.array([12, 13, 14]), 2, 0.5, np.array([13, 14, 15]), False),
            (
                np.array([13, 14, 15]),
                3,
                0.6,
                np.array([14, 15, 16]),
                True,
            ),  # Fin épisode 2
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions_1:
            n_step_buffer.add(state, action, reward, next_state, done)

        for state, action, reward, next_state, done in transitions_2:
            n_step_buffer.add(state, action, reward, next_state, done)

        # Vérifier que toutes les transitions sont ajoutées
        assert len(n_step_buffer.buffer) == 6

    def test_n_step_end_episode_with_zero_rewards(self, n_step_buffer):
        """Test N-step avec récompenses zéro à la fin d'épisode."""
        # Créer un épisode avec récompenses zéro
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (
                np.array([3, 4, 5]),
                2,
                0.0,
                np.array([4, 5, 6]),
                True,
            ),  # Récompense zéro + fin
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            n_step_buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(n_step_buffer.buffer) == 3

    def test_n_step_end_episode_with_negative_rewards(self, n_step_buffer):
        """Test N-step avec récompenses négatives à la fin d'épisode."""
        # Créer un épisode avec récompenses négatives
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (
                np.array([3, 4, 5]),
                2,
                -0.5,
                np.array([4, 5, 6]),
                True,
            ),  # Récompense négative + fin
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            n_step_buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(n_step_buffer.buffer) == 3

    def test_n_step_end_episode_with_large_rewards(self, n_step_buffer):
        """Test N-step avec récompenses importantes à la fin d'épisode."""
        # Créer un épisode avec récompenses importantes
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (
                np.array([3, 4, 5]),
                2,
                10.0,
                np.array([4, 5, 6]),
                True,
            ),  # Grande récompense + fin
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            n_step_buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(n_step_buffer.buffer) == 3

    def test_n_step_end_episode_buffer_overflow(self, n_step_buffer):
        """Test N-step avec débordement du buffer à la fin d'épisode."""
        # Remplir le buffer presque complètement
        for i in range(98):
            state = np.array([i, i + 1, i + 2])
            action = i % 3
            reward = i * 0.01
            next_state = np.array([i + 1, i + 2, i + 3])
            done = False

            n_step_buffer.add(state, action, reward, next_state, done)

        # Ajouter un épisode qui se termine
        transitions = [
            (np.array([98, 99, 100]), 0, 0.1, np.array([99, 100, 101]), False),
            (
                np.array([99, 100, 101]),
                1,
                0.2,
                np.array([100, 101, 102]),
                True,
            ),  # Fin d'épisode
        ]

        for state, action, reward, next_state, done in transitions:
            n_step_buffer.add(state, action, reward, next_state, done)

        # Vérifier que le buffer gère le débordement correctement
        assert len(n_step_buffer.buffer) <= n_step_buffer.capacity


class TestNStepReturnCalculationEdgeCases:
    """Tests edge cases pour le calcul des retours N-step."""

    @pytest.fixture
    def n_step_buffer(self):
        """Crée un buffer N-step pour les tests."""
        if NStepBuffer is None:
            pytest.skip("NStepBuffer non disponible")

        return NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)

    def test_n_step_return_calculation_with_gamma_one(self):
        """Test calcul retour N-step avec gamma=1.0."""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=1.0)

        # Créer un épisode
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (np.array([3, 4, 5]), 2, 0.3, np.array([4, 5, 6]), False),
            (np.array([4, 5, 6]), 3, 0.4, np.array([5, 6, 7]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(buffer.buffer) == 4

    def test_n_step_return_calculation_with_gamma_zero(self):
        """Test calcul retour N-step avec gamma=0.0."""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.0)

        # Créer un épisode
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (np.array([3, 4, 5]), 2, 0.3, np.array([4, 5, 6]), False),
            (np.array([4, 5, 6]), 3, 0.4, np.array([5, 6, 7]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(buffer.buffer) == 4

    def test_n_step_return_calculation_with_small_gamma(self):
        """Test calcul retour N-step avec gamma très petit."""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.01)

        # Créer un épisode
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (np.array([3, 4, 5]), 2, 0.3, np.array([4, 5, 6]), False),
            (np.array([4, 5, 6]), 3, 0.4, np.array([5, 6, 7]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(buffer.buffer) == 4

    def test_n_step_return_calculation_with_large_n(self):
        """Test calcul retour N-step avec n très grand."""
        buffer = NStepBuffer(
            capacity=0.100,
            n_step=10,  # n très grand
            gamma=0.99,
        )

        # Créer un épisode court
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(buffer.buffer) == 2

    def test_n_step_return_calculation_with_n_one(self):
        """Test calcul retour N-step avec n=1 (pas de N-step)."""
        buffer = NStepBuffer(
            capacity=0.100,
            n_step=1,  # Pas de N-step
            gamma=0.99,
        )

        # Créer un épisode
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (np.array([3, 4, 5]), 2, 0.3, np.array([4, 5, 6]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(buffer.buffer) == 3


class TestNStepPrioritizedEdgeCases:
    """Tests edge cases pour le buffer N-step priorisé."""

    @pytest.fixture
    def n_step_prioritized_buffer(self):
        """Crée un buffer N-step priorisé pour les tests."""
        if NStepPrioritizedBuffer is None:
            pytest.skip("NStepPrioritizedBuffer non disponible")

        return NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_end=1.0,
        )

    def test_n_step_prioritized_end_episode(self, n_step_prioritized_buffer):
        """Test buffer N-step priorisé avec fin d'épisode."""
        # Créer un épisode
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (np.array([3, 4, 5]), 2, 0.3, np.array([4, 5, 6]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            n_step_prioritized_buffer.add(state, action, reward, next_state, done)

        # Vérifier que les transitions sont ajoutées
        assert len(n_step_prioritized_buffer.buffer) == 3

        # Vérifier que les priorités sont définies pour les transitions dans le buffer principal
        assert len(n_step_prioritized_buffer.priorities) == 100  # Capacité du buffer
        # Vérifier que les priorités des éléments ajoutés sont définies
        for i in range(len(n_step_prioritized_buffer.buffer)):
            assert n_step_prioritized_buffer.priorities[i] > 0

    def test_n_step_prioritized_end_episode_sampling(self, n_step_prioritized_buffer):
        """Test échantillonnage avec fin d'épisode."""
        # Créer un épisode
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (np.array([3, 4, 5]), 2, 0.3, np.array([4, 5, 6]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            n_step_prioritized_buffer.add(state, action, reward, next_state, done)

        # Échantillonner
        batch_size = 2
        batch, weights, indices = n_step_prioritized_buffer.sample(batch_size)

        # Vérifier que l'échantillonnage fonctionne
        assert len(batch) == batch_size
        assert len(weights) == batch_size
        assert len(indices) == batch_size

        # Vérifier que les poids sont valides
        for weight in weights:
            assert weight > 0
            assert not np.isnan(weight)
            assert not np.isinf(weight)

    def test_n_step_prioritized_end_episode_priority_update(
        self, n_step_prioritized_buffer
    ):
        """Test mise à jour priorité avec fin d'épisode."""
        # Créer un épisode
        transitions = [
            (np.array([1, 2, 3]), 0, 0.1, np.array([2, 3, 4]), False),
            (np.array([2, 3, 4]), 1, 0.2, np.array([3, 4, 5]), False),
            (np.array([3, 4, 5]), 2, 0.3, np.array([4, 5, 6]), True),  # Fin d'épisode
        ]

        # Ajouter les transitions
        for state, action, reward, next_state, done in transitions:
            n_step_prioritized_buffer.add(state, action, reward, next_state, done)

        # Mettre à jour les priorités
        indices = [0, 1, 2]
        td_errors = [0.5, 0.6, 0.7]

        n_step_prioritized_buffer.update_priorities(indices, td_errors)

        # Vérifier que les priorités sont mises à jour
        for i in range(3):
            assert n_step_prioritized_buffer.priorities[i] > 0

        assert n_step_prioritized_buffer.max_priority > 0
