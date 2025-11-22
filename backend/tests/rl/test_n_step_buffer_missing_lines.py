"""Tests supplémentaires pour n_step_buffer.py - lignes manquantes."""

from unittest.mock import patch

import numpy as np

from services.rl.n_step_buffer import NStepBuffer, NStepPrioritizedBuffer


class TestNStepBufferMissingLines:
    """Tests pour couvrir les lignes manquantes de NStepBuffer."""

    def test_add_transition_exception(self):
        """Test add_transition avec exception."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Mock logger pour vérifier l'erreur
        with patch.object(buffer.logger, "error") as mock_error:
            # Créer une transition invalide pour déclencher une exception
            with patch.object(buffer, "_process_n_step_transitions", side_effect=Exception("Test error")):
                buffer.add_transition(
                    state=np.array([1, 2, 3]), action=1, reward=1, next_state=np.array([4, 5, 6]), done=False, info=None
                )

            # Vérifier que l'erreur a été loggée
            mock_error.assert_called_once()

    def test_add_method_compatibility(self):
        """Test méthode add de compatibilité."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Test avec la méthode add
        buffer.add(
            state=np.array([1, 2, 3]),
            action=1,
            reward=1,
            next_state=np.array([4, 5, 6]),
            done=False,
            info={"test": "info"},
        )

        # Vérifier que la transition a été ajoutée au buffer temporaire
        assert len(buffer.temp_buffer) == 1

    def test_process_n_step_transitions_empty_temp_buffer(self):
        """Test _process_n_step_transitions avec buffer temporaire vide."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer temporaire vide
        buffer.temp_buffer = []

        # Ne devrait pas lever d'exception
        buffer._process_n_step_transitions()

        # Buffer principal devrait rester vide
        assert len(buffer.buffer) == 0

    def test_process_n_step_transitions_exception(self):
        """Test _process_n_step_transitions avec exception."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Ajouter une transition au buffer temporaire
        buffer.temp_buffer = [
            {
                "state": np.array([1, 2, 3]),
                "action": 1,
                "reward": 1,
                "next_state": np.array([4, 5, 6]),
                "done": False,
                "info": None,
            }
        ]

        # Mock _calculate_n_step_return pour lever une exception
        with (
            patch.object(buffer, "_calculate_n_step_return", side_effect=Exception("Test error")),
            patch.object(buffer.logger, "error") as mock_error,
        ):
            buffer._process_n_step_transitions()

            # Vérifier que l'erreur a été loggée
            mock_error.assert_called_once()

    def test_calculate_n_step_return_with_nan_reward(self):
        """Test _calculate_n_step_return avec récompense NaN."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompense NaN
        buffer.temp_buffer = [{"reward": np.nan}, {"reward": 1}, {"reward": 2}]

        # Ne devrait pas lever d'exception
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)

    def test_calculate_n_step_return_with_inf_reward(self):
        """Test _calculate_n_step_return avec récompense infinie."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompense infinie
        buffer.temp_buffer = [{"reward": np.inf}, {"reward": 1}, {"reward": 2}]

        # Ne devrait pas lever d'exception
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)

    def test_calculate_n_step_return_with_neg_inf_reward(self):
        """Test _calculate_n_step_return avec récompense négative infinie."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompense négative infinie
        buffer.temp_buffer = [{"reward": -np.inf}, {"reward": 1}, {"reward": 2}]

        # Ne devrait pas lever d'exception
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)

    def test_get_final_next_state_exception(self):
        """Test _get_final_next_state avec exception."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec données invalides
        buffer.temp_buffer = [
            {"next_state": "invalid_state"}  # String au lieu de numpy array
        ]

        # Ne devrait pas lever d'exception
        result = buffer._get_final_next_state(0)
        assert result is not None

    def test_sample_empty_buffer(self):
        """Test sample avec buffer vide."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer vide
        batch = buffer.sample(5)

        assert batch == ([], [])

    def test_sample_with_exception(self):
        """Test sample avec exception."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Ajouter quelques transitions
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=False,
                info=None,
            )

        # Mock random.choices pour lever une exception
        with (
            patch("random.choices", side_effect=Exception("Test error")),
            patch.object(buffer.logger, "error") as mock_error,
        ):
            batch = buffer.sample(2)

            # Devrait retourner des listes vides en cas d'erreur
            assert batch == ([], [])

            # Vérifier que l'erreur a été loggée
            mock_error.assert_called_once()

    def test_get_stats(self):
        """Test get_stats."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Ajouter quelques transitions
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=False,
                info=None,
            )

        stats = buffer.get_stats()

        assert isinstance(stats, dict)
        assert "buffer_size" in stats
        assert "temp_buffer_size" in stats
        assert "capacity" in stats
        assert "n_step" in stats

    def test_get_stats_with_exception(self):
        """Test get_stats avec exception."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Mock len pour lever une exception
        with (
            patch("builtins.len", side_effect=Exception("Test error")),
            patch.object(buffer.logger, "error") as mock_error,
        ):
            stats = buffer.get_stats()

            # Devrait retourner des stats par défaut
            assert isinstance(stats, dict)

            # Vérifier que l'erreur a été loggée
            mock_error.assert_called_once()

    def test_clear(self):
        """Test clear."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Ajouter quelques transitions
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=False,
                info=None,
            )

        # Vider le buffer
        buffer.clear()

        assert len(buffer.buffer) == 0
        assert len(buffer.temp_buffer) == 0


class TestNStepPrioritizedBufferMissingLines:
    """Tests pour couvrir les lignes manquantes de NStepPrioritizedBuffer."""

    def test_add_transition_with_td_error(self):
        """Test add_transition avec td_error."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Ajouter une transition avec td_error
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=1,
            next_state=np.array([4, 5, 6]),
            done=False,
            info=None,
            td_error=2,
        )

        # Vérifier que la priorité a été mise à jour
        assert buffer.max_priority >= 2

    def test_add_transition_without_td_error(self):
        """Test add_transition sans td_error."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Ajouter une transition sans td_error
        buffer.add_transition(
            state=np.array([1, 2, 3]), action=1, reward=1, next_state=np.array([4, 5, 6]), done=False, info=None
        )

        # Vérifier que la priorité par défaut a été utilisée
        assert buffer.max_priority >= 1

    def test_sample_empty_buffer(self):
        """Test sample avec buffer vide."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Buffer vide
        batch, weights, indices = buffer.sample(5)

        assert batch == []
        assert weights == []
        assert indices == []

    def test_sample_with_exception(self):
        """Test sample avec exception."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Ajouter quelques transitions
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=False,
                info=None,
            )

        # Mock random.choices pour lever une exception
        with (
            patch("random.choices", side_effect=Exception("Test error")),
            patch.object(buffer.logger, "error") as mock_error,
        ):
            batch, weights, indices = buffer.sample(2)

            # Devrait retourner des listes vides en cas d'erreur
            assert batch == []
            assert weights == []
            assert indices == []

            # Vérifier que l'erreur a été loggée
            mock_error.assert_called_once()

    def test_update_priorities(self):
        """Test update_priorities."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Ajouter quelques transitions
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=False,
                info=None,
            )

        # Mettre à jour les priorités
        indices = [0, 1, 2]
        priorities = [2, 3, 4]

        buffer.update_priorities(indices, priorities)

        # Vérifier que max_priority a été mis à jour
        assert buffer.max_priority >= 4

    def test_clear(self):
        """Test clear."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Ajouter quelques transitions
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=False,
                info=None,
            )

        # Vider le buffer
        buffer.clear()

        assert len(buffer.buffer) == 0
        assert len(buffer.temp_buffer) == 0
        assert len(buffer.priorities) == 0
        assert buffer.max_priority == 1
        assert buffer.beta == buffer.beta_start

    def test_get_stats(self):
        """Test get_stats."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Ajouter quelques transitions
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=False,
                info=None,
            )

        stats = buffer.get_stats()

        assert isinstance(stats, dict)
        assert "buffer_size" in stats
        assert "temp_buffer_size" in stats
        assert "capacity" in stats
        assert "n_step" in stats
        assert "max_priority" in stats
        assert "beta" in stats

    def test_get_stats_with_exception(self):
        """Test get_stats avec exception."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Mock len pour lever une exception
        with (
            patch("builtins.len", side_effect=Exception("Test error")),
            patch.object(buffer.logger, "error") as mock_error,
        ):
            stats = buffer.get_stats()

            # Devrait retourner des stats par défaut
            assert isinstance(stats, dict)

            # Vérifier que l'erreur a été loggée
            mock_error.assert_called_once()

    def test_n_step_calculation_with_nan_rewards(self):
        """Test calcul N-step avec récompenses NaN."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompenses NaN
        buffer.temp_buffer = [{"reward": np.nan}, {"reward": np.nan}, {"reward": np.nan}]

        # Ne devrait pas lever d'exception
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)

    def test_n_step_calculation_with_inf_rewards(self):
        """Test calcul N-step avec récompenses infinies."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompenses infinies
        buffer.temp_buffer = [{"reward": np.inf}, {"reward": -np.inf}, {"reward": np.inf}]

        # Ne devrait pas lever d'exception
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)

    def test_capacity_overflow(self):
        """Test débordement de capacité."""
        buffer = NStepPrioritizedBuffer(capacity=3, n_step=2)

        # Ajouter plus de transitions que la capacité
        for i in range(5):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2]),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5]),
                done=False,
                info=None,
            )

        # Le buffer ne devrait pas dépasser la capacité
        assert len(buffer.buffer) <= buffer.capacity

    def test_empty_temp_buffer_edge_cases(self):
        """Test cas limites avec buffer temporaire vide."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Buffer temporaire vide
        buffer.temp_buffer = []

        # Ces méthodes ne devraient pas lever d'exception
        result1 = buffer._calculate_n_step_return(0)
        result2 = buffer._get_final_next_state(0)

        assert isinstance(result1, float)
        assert result2 is not None

    def test_negative_rewards(self):
        """Test avec récompenses négatives."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompenses négatives
        buffer.temp_buffer = [{"reward": -1}, {"reward": -2}, {"reward": -3}]

        # Ne devrait pas lever d'exception
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)
        assert result < 0  # Résultat devrait être négatif
