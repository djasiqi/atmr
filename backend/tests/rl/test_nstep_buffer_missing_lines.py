#!/usr/bin/env python3
"""
Tests supplémentaires pour couvrir les lignes manquantes dans n_step_buffer.py
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from services.rl.n_step_buffer import NStepBuffer, NStepPrioritizedBuffer


class TestNStepBufferMissingLines:
    """Tests pour couvrir les lignes manquantes dans NStepBuffer."""

    def test_add_transition_with_none_info(self):
        """Test ligne 96-97: add_transition avec info=None."""
        buffer = NStepBuffer(capacity=10, n_step=3, gamma=0.9)

        buffer.add_transition(
            state=np.array([1, 2, 3]), action=0, reward=1.0, next_state=np.array([4, 5, 6]), done=False, info=None
        )

        assert len(buffer.temp_buffer) == 1
        # info=None est converti en {} dans le code
        assert buffer.temp_buffer[0]["info"] == {}

    def test_calculate_n_step_return_with_n_step_one(self):
        """Test ligne 119: _calculate_n_step_return avec n_step=1."""
        buffer = NStepBuffer(capacity=10, n_step=1, gamma=0.9)
        buffer.temp_buffer = [{"reward": 1.0, "done": False}, {"reward": 2.0, "done": True}]

        result = buffer._calculate_n_step_return(0)
        assert result == 1.0  # Pas de discount avec n_step=1


class TestNStepPrioritizedBufferMissingLines:
    """Tests pour couvrir les lignes manquantes dans NStepPrioritizedBuffer."""

    def test_add_transition_with_none_td_error(self):
        """Test lignes 232-234: add_transition avec td_error=None."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3, gamma=0.9)

        # Ajouter 3 transitions pour compléter n_step
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i + 1),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=i == 2,
                info=None,
                td_error=None,
            )

        assert len(buffer.buffer) == 3  # Trois transitions ajoutées
        assert buffer.priorities[0] > 0  # Priorité basée sur reward

    def test_add_transition_with_td_error(self):
        """Test ligne 238: add_transition avec td_error défini."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3, gamma=0.9)

        # Ajouter 3 transitions pour compléter n_step
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i + 1),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=i == 2,
                info=None,
                td_error=float(i + 1),
            )

        assert len(buffer.buffer) == 3  # Trois transitions ajoutées
        assert buffer.priorities[0] > 0  # Priorité basée sur td_error

    def test_sample_empty_buffer(self):
        """Test lignes 272-274: sample avec buffer vide."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3, gamma=0.9)

        batch, weights, indices = buffer.sample(5)

        assert len(batch) == 0
        assert len(weights) == 0
        assert len(indices) == 0

    def test_sample_with_probabilities(self):
        """Test lignes 349-358: sample avec probabilités."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3, gamma=0.9)

        # Ajouter plusieurs transitions
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i + 1),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=i == 2,
                info=None,
                td_error=float(i + 1),
            )

        batch, weights, indices = buffer.sample(2)

        assert len(batch) == 2
        assert len(weights) == 2
        assert len(indices) == 2
        assert all(0 <= w <= 1 for w in weights)

    def test_sample_with_exception(self):
        """Test lignes 362-370: sample avec exception."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3, gamma=0.9)

        # Créer des données invalides pour déclencher une exception
        buffer.buffer = [{"invalid": "data"}]
        buffer.priorities = np.array([float("nan")])

        batch, weights, indices = buffer.sample(1)

        assert len(batch) == 0
        assert len(weights) == 0
        assert len(indices) == 0

    def test_get_stats(self):
        """Test ligne 394: get_stats."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3, gamma=0.9)

        # Ajouter quelques transitions complètes
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i + 1),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=i == 2,
                info=None,
                td_error=float(i + 1),
            )

        stats = buffer.get_stats()

        assert "buffer_size" in stats
        assert "alpha" in stats
        assert "beta_start" in stats
        assert "beta_end" in stats
        assert "max_priority" in stats
        assert stats["buffer_size"] == 3  # Trois transitions ajoutées

    def test_clear(self):
        """Test lignes 443-444: clear."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3, gamma=0.9)

        # Ajouter des données complètes
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i + 1),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=i == 2,
                info=None,
                td_error=float(i + 1),
            )

        assert len(buffer.buffer) == 3  # Trois transitions ajoutées

        buffer.clear()

        assert len(buffer.buffer) == 0
        assert len(buffer.priorities) == 10  # Le tableau garde sa taille mais est remis à zéro
        assert buffer.max_priority == 1.0

    def test_calculate_n_step_return_with_nan(self):
        """Test lignes 457-459: _calculate_n_step_return avec NaN."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3, gamma=0.9)
        buffer.temp_buffer = [{"reward": float("nan"), "done": False}, {"reward": 2.0, "done": True}]

        result = buffer._calculate_n_step_return(0)
        # Le calcul inclut les deux transitions: NaN (0.0) + 2.0*0.9 = 1.8
        assert result == 1.8

    def test_calculate_n_step_return_with_inf(self):
        """Test lignes 483-486: _calculate_n_step_return avec inf."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3, gamma=0.9)
        buffer.temp_buffer = [{"reward": float("inf"), "done": False}, {"reward": 2.0, "done": True}]

        result = buffer._calculate_n_step_return(0)
        # Le calcul inclut les deux transitions: inf (1.0) + 2.0*0.9 = 2.8
        assert result == 2.8

    def test_calculate_n_step_return_with_neg_inf(self):
        """Test lignes 483-486: _calculate_n_step_return avec -inf."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3, gamma=0.9)
        buffer.temp_buffer = [{"reward": float("-inf"), "done": False}, {"reward": 2.0, "done": True}]

        result = buffer._calculate_n_step_return(0)
        # Le calcul inclut les deux transitions: -inf (-1.0) + 2.0*0.9 = 0.8
        assert result == 0.8
