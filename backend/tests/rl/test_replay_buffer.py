# ruff: noqa: DTZ001, DTZ003
"""
Tests pour le Replay Buffer.

Teste:
- Ajout de transitions
- Échantillonnage
- Gestion de la capacité
- Statistiques
"""
import numpy as np
import pytest

from services.rl.replay_buffer import ReplayBuffer, Transition


class TestReplayBufferBasics:
    """Tests basiques du buffer."""

    def test_buffer_creation(self):
        """Test création du buffer."""
        buffer = ReplayBuffer(capacity=1000)

        assert len(buffer) == 0
        assert buffer.capacity == 1000

    def test_buffer_push_single(self):
        """Test ajout d'une transition."""
        buffer = ReplayBuffer(capacity=1000)

        state = np.random.rand(122)
        next_state = np.random.rand(122)

        buffer.push(state, 1, next_state, 50.0, False)

        assert len(buffer) == 1

    def test_buffer_push_multiple(self):
        """Test ajout de plusieurs transitions."""
        buffer = ReplayBuffer(capacity=1000)

        for i in range(100):
            state = np.random.rand(122)
            next_state = np.random.rand(122)
            buffer.push(state, i, next_state, float(i), False)

        assert len(buffer) == 100

    def test_buffer_capacity_overflow(self):
        """Test que le buffer respecte la capacité maximale."""
        buffer = ReplayBuffer(capacity=10)

        # Ajouter 20 transitions
        for i in range(20):
            buffer.push(np.random.rand(122), i, np.random.rand(122), 1.0, False)

        # Ne devrait garder que les 10 dernières
        assert len(buffer) == 10

    def test_buffer_fifo_order(self):
        """Test que le buffer est FIFO (First In First Out)."""
        buffer = ReplayBuffer(capacity=5)

        # Ajouter 10 transitions avec rewards uniques
        for i in range(10):
            buffer.push(np.random.rand(122), i, np.random.rand(122), float(i), False)

        # Les 5 dernières (rewards 5, 6, 7, 8, 9) devraient être conservées
        rewards = [t.reward for t in buffer.buffer]
        assert rewards == [5.0, 6.0, 7.0, 8.0, 9.0]


class TestReplayBufferSampling:
    """Tests de l'échantillonnage."""

    def test_buffer_sample(self):
        """Test échantillonnage de base."""
        buffer = ReplayBuffer(capacity=1000)

        # Remplir avec 100 transitions
        for i in range(100):
            buffer.push(np.random.rand(122), i, np.random.rand(122), 1.0, False)

        # Échantillonner 32 transitions
        batch = buffer.sample(32)

        assert len(batch) == 32
        assert all(isinstance(t, Transition) for t in batch)

    def test_buffer_sample_randomness(self):
        """Test que l'échantillonnage est aléatoire."""
        buffer = ReplayBuffer(capacity=1000)

        # Remplir avec transitions identifiables
        for i in range(100):
            buffer.push(np.random.rand(122), i, np.random.rand(122), float(i), False)

        # Deux échantillonnages devraient être différents
        batch1 = buffer.sample(20)
        batch2 = buffer.sample(20)

        actions1 = [t.action for t in batch1]
        actions2 = [t.action for t in batch2]

        assert actions1 != actions2  # Très probable

    def test_buffer_sample_too_large(self):
        """Test qu'échantillonner plus que la taille lève une erreur."""
        buffer = ReplayBuffer(capacity=1000)

        # Ajouter seulement 10 transitions
        for i in range(10):
            buffer.push(np.random.rand(122), i, np.random.rand(122), 1.0, False)

        # Essayer d'échantillonner 20 → devrait lever une erreur
        with pytest.raises(ValueError):
            buffer.sample(20)

    def test_buffer_is_ready(self):
        """Test de la méthode is_ready()."""
        buffer = ReplayBuffer(capacity=1000)

        assert not buffer.is_ready(min_size=64)

        # Ajouter 50 transitions
        for i in range(50):
            buffer.push(np.random.rand(122), i, np.random.rand(122), 1.0, False)

        assert not buffer.is_ready(min_size=64)  # Encore pas prêt

        # Ajouter 20 de plus (total 70)
        for i in range(20):
            buffer.push(np.random.rand(122), i, np.random.rand(122), 1.0, False)

        assert buffer.is_ready(min_size=64)  # Maintenant prêt


class TestReplayBufferUtilities:
    """Tests des fonctions utilitaires."""

    def test_buffer_clear(self):
        """Test vidage du buffer."""
        buffer = ReplayBuffer(capacity=1000)

        # Remplir
        for i in range(50):
            buffer.push(np.random.rand(122), i, np.random.rand(122), 1.0, False)

        assert len(buffer) == 50

        # Vider
        buffer.clear()

        assert len(buffer) == 0

    def test_buffer_get_latest(self):
        """Test récupération des dernières transitions."""
        buffer = ReplayBuffer(capacity=1000)

        # Ajouter 10 transitions
        for i in range(10):
            buffer.push(np.random.rand(122), i, np.random.rand(122), float(i), False)

        # Récupérer les 3 dernières
        latest = buffer.get_latest(n=3)

        assert len(latest) == 3
        rewards = [t.reward for t in latest]
        assert rewards == [7.0, 8.0, 9.0]

    def test_buffer_get_latest_more_than_size(self):
        """Test get_latest avec n > taille."""
        buffer = ReplayBuffer(capacity=1000)

        # Ajouter seulement 5 transitions
        for i in range(5):
            buffer.push(np.random.rand(122), i, np.random.rand(122), 1.0, False)

        # Demander 10 → devrait retourner seulement 5
        latest = buffer.get_latest(n=10)

        assert len(latest) == 5

    def test_buffer_statistics_empty(self):
        """Test statistiques sur buffer vide."""
        buffer = ReplayBuffer(capacity=1000)

        stats = buffer.get_statistics()

        assert stats["size"] == 0
        assert stats["avg_reward"] == 0.0

    def test_buffer_statistics(self):
        """Test calcul de statistiques."""
        buffer = ReplayBuffer(capacity=1000)

        # Ajouter transitions avec rewards variées
        rewards = [10.0, 20.0, 30.0, 40.0, 50.0, -10.0, -20.0]
        for i, r in enumerate(rewards):
            buffer.push(np.random.rand(122), i, np.random.rand(122), r, i % 2 == 0)

        stats = buffer.get_statistics()

        assert stats["size"] == 7
        assert stats["avg_reward"] == pytest.approx(np.mean(rewards))
        assert stats["std_reward"] == pytest.approx(np.std(rewards))
        assert stats["min_reward"] == -20.0
        assert stats["max_reward"] == 50.0
        assert stats["done_ratio"] == pytest.approx(4/7)  # 4 True sur 7

