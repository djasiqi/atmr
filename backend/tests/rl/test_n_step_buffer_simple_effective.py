"""
Tests simples et efficaces pour améliorer la couverture de n_step_buffer.py
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from services.rl.n_step_buffer import NStepBuffer, NStepPrioritizedBuffer


class TestNStepBufferSimple:
    """Tests simples pour NStepBuffer"""

    def test_init_basic(self):
        """Test initialisation basique"""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)
        assert buffer.capacity == 100
        assert buffer.n_step == 3
        assert buffer.gamma == 0.99
        assert len(buffer.buffer) == 0

    def test_add_transition_basic(self):
        """Test ajout de transition basique"""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=10.0,
            next_state=np.array([4, 5, 6]),
            done=False
        )
        # Le buffer N-step stocke temporairement les transitions
        assert len(buffer.temp_buffer) == 1

    def test_sample_basic(self):
        """Test échantillonnage basique"""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)

        # Ajouter quelques transitions
        for i in range(5):
            buffer.add_transition(
                state=np.array([i, i+1, i+2]),
                action=i,
                reward=float(i),
                next_state=np.array([i+3, i+4, i+5]),
                done=False
            )

        batch, indices = buffer.sample(batch_size=3)
        assert len(batch) == 3
        assert len(indices) == 3

    def test_sample_empty(self):
        """Test échantillonnage avec buffer vide"""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)
        batch, indices = buffer.sample(batch_size=3)
        assert len(batch) == 0
        assert len(indices) == 0

    def test_get_stats(self):
        """Test statistiques"""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)
        stats = buffer.get_stats()
        assert isinstance(stats, dict)
        assert "buffer_size" in stats
        assert "capacity" in stats

    def test_clear(self):
        """Test nettoyage du buffer"""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=10.0,
            next_state=np.array([4, 5, 6]),
            done=False
        )
        buffer.clear()
        assert len(buffer.buffer) == 0


class TestNStepPrioritizedBufferSimple:
    """Tests simples pour NStepPrioritizedBuffer"""

    def test_init_basic(self):
        """Test initialisation basique"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )
        assert buffer.capacity == 100
        assert buffer.n_step == 3
        assert buffer.gamma == 0.99
        assert buffer.alpha == 0.6
        assert buffer.beta_start == 0.4
        assert len(buffer.buffer) == 0

    def test_add_transition_basic(self):
        """Test ajout de transition basique"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=10.0,
            next_state=np.array([4, 5, 6]),
            done=False
        )
        # Le buffer N-step stocke temporairement les transitions
        assert len(buffer.temp_buffer) == 1

    def test_sample_basic(self):
        """Test échantillonnage basique"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )

        # Ajouter quelques transitions
        for i in range(5):
            buffer.add_transition(
                state=np.array([i, i+1, i+2]),
                action=i,
                reward=float(i),
                next_state=np.array([i+3, i+4, i+5]),
                done=False
            )

        batch, weights, indices = buffer.sample(batch_size=3)
        assert len(batch) == 3
        assert len(weights) == 3
        assert len(indices) == 3

    def test_sample_empty(self):
        """Test échantillonnage avec buffer vide"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )
        batch, weights, indices = buffer.sample(batch_size=3)
        assert len(batch) == 0
        assert len(weights) == 0
        assert len(indices) == 0

    def test_update_priorities(self):
        """Test mise à jour des priorités"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )

        # Ajouter quelques transitions et forcer le calcul N-step
        for i in range(5):  # Plus que n_step pour déclencher le calcul
            buffer.add_transition(
                state=np.array([i, i+1, i+2]),
                action=i,
                reward=float(i),
                next_state=np.array([i+3, i+4, i+5]),
                done=False
            )

        # Mettre à jour les priorités
        buffer.update_priorities([0, 1, 2], [0.1, 0.2, 0.3])
        assert len(buffer.priorities) == 100  # Taille fixe du tableau de priorités

    def test_get_stats(self):
        """Test statistiques"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )
        stats = buffer.get_stats()
        assert isinstance(stats, dict)
        assert "buffer_size" in stats
        assert "capacity" in stats

    def test_clear(self):
        """Test nettoyage du buffer"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=10.0,
            next_state=np.array([4, 5, 6]),
            done=False
        )
        buffer.clear()
        assert len(buffer.buffer) == 0
        assert len(buffer.priorities) == 100  # Taille fixe du tableau


class TestNStepBufferEdgeCases:
    """Tests pour cas limites"""

    def test_n_step_one(self):
        """Test avec n_step=1"""
        buffer = NStepBuffer(capacity=0.100, n_step=1, gamma=0.99)
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=10.0,
            next_state=np.array([4, 5, 6]),
            done=False
        )
        batch, _indices = buffer.sample(batch_size=1)
        assert len(batch) == 1

    def test_done_transition(self):
        """Test avec transition terminée"""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=10.0,
            next_state=np.array([4, 5, 6]),
            done=True
        )
        batch, _indices = buffer.sample(batch_size=1)
        assert len(batch) == 1

    def test_capacity_overflow(self):
        """Test dépassement de capacité"""
        buffer = NStepBuffer(capacity=3, n_step=2, gamma=0.99)

        # Ajouter plus que la capacité
        for i in range(5):
            buffer.add_transition(
                state=np.array([i, i+1, i+2]),
                action=i,
                reward=float(i),
                next_state=np.array([i+3, i+4, i+5]),
                done=False
            )

        # Le buffer ne devrait pas dépasser la capacité
        assert len(buffer.buffer) <= buffer.capacity

    def test_negative_rewards(self):
        """Test avec récompenses négatives"""
        buffer = NStepBuffer(capacity=0.100, n_step=3, gamma=0.99)
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=-10.0,
            next_state=np.array([4, 5, 6]),
            done=False
        )
        # Ajouter plus de transitions pour déclencher le calcul N-step
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i+1, i+2]),
                action=i,
                reward=float(i),
                next_state=np.array([i+3, i+4, i+5]),
                done=False
            )
        batch, _indices = buffer.sample(batch_size=1)
        assert len(batch) == 1


class TestNStepPrioritizedBufferEdgeCases:
    """Tests pour cas limites du buffer priorisé"""

    def test_zero_priority(self):
        """Test avec priorité zéro"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )
        # Ajouter des transitions pour avoir des priorités
        for i in range(5):
            buffer.add_transition(
                state=np.array([i, i+1, i+2]),
                action=i,
                reward=float(i),
                next_state=np.array([i+3, i+4, i+5]),
                done=False
            )
        buffer.update_priorities([0], [0.0])
        assert buffer.priorities[0] > 0  # Devrait être corrigé à epsilon

    def test_negative_priority(self):
        """Test avec priorité négative"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )
        # Ajouter des transitions pour avoir des priorités
        for i in range(5):
            buffer.add_transition(
                state=np.array([i, i+1, i+2]),
                action=i,
                reward=float(i),
                next_state=np.array([i+3, i+4, i+5]),
                done=False
            )
        buffer.update_priorities([0], [-0.1])
        assert buffer.priorities[0] > 0  # Devrait être corrigé à epsilon

    def test_nan_priority(self):
        """Test avec priorité NaN"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=10.0,
            next_state=np.array([4, 5, 6]),
            done=False
        )
        buffer.update_priorities([0], [float("nan")])
        assert not np.isnan(buffer.priorities[0])  # Devrait être corrigé

    def test_inf_priority(self):
        """Test avec priorité infinie"""
        buffer = NStepPrioritizedBuffer(
            capacity=0.100,
            n_step=3,
            gamma=0.99,
            alpha=0.6,
            beta_start=0.4,
            beta_increment=0.0001
        )
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=10.0,
            next_state=np.array([4, 5, 6]),
            done=False
        )
        buffer.update_priorities([0], [float("inf")])
        assert not np.isinf(buffer.priorities[0])  # Devrait être corrigé
