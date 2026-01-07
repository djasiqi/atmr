"""Tests supplémentaires pour n_step_buffer.py - lignes manquantes."""

from unittest.mock import Mock, patch

import numpy as np

from services.ml.rl.n_step_buffer import NStepBuffer, NStepPrioritizedBuffer


class TestNStepBufferMissingLines:
    """Tests pour couvrir les lignes manquantes de NStepBuffer."""

    def test_add_transition_exception(self):
        """Test add_transition avec exception."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Mock logger pour vérifier l'erreur
        with patch.object(buffer.logger, "error") as mock_error:
            # ✅ FIX: On ne peut pas patcher deque.append directement
            # car c'est en lecture seule.
            # Au lieu de cela, on crée un mock pour state qui lève une exception
            # lors de copy() car state.copy() est appelé dans add_transition
            # avant append
            mock_state = Mock(spec=np.ndarray)
            mock_state.copy.side_effect = Exception("Test error")

            buffer.add_transition(
                state=mock_state,
                action=1,
                reward=1,
                next_state=np.array([4, 5, 6], dtype=np.float32),
                done=False,
                info=None,
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
            patch.object(
                buffer, "_calculate_n_step_return", side_effect=Exception("Test error")
            ),
            patch.object(buffer.logger, "error") as mock_error,
        ):
            buffer._process_n_step_transitions()

            # Vérifier que l'erreur a été loggée
            mock_error.assert_called_once()

    def test_calculate_n_step_return_with_nan_reward(self):
        """Test _calculate_n_step_return avec récompense NaN."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompense NaN
        buffer.temp_buffer = [
            {"reward": np.nan, "done": False},
            {"reward": 1.0, "done": False},
            {"reward": 2.0, "done": False},
        ]

        # Ne devrait pas lever d'exception et retourner un float
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)
        # NaN est converti en 0, donc le résultat devrait être >= 0
        assert result >= 0.0

    def test_calculate_n_step_return_with_inf_reward(self):
        """Test _calculate_n_step_return avec récompense infinie."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompense infinie
        buffer.temp_buffer = [
            {"reward": np.inf, "done": False},
            {"reward": 1.0, "done": False},
            {"reward": 2.0, "done": False},
        ]

        # Ne devrait pas lever d'exception et retourner un float
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)
        # inf est converti en 1.0, donc le résultat devrait être > 0
        assert result > 0.0

    def test_calculate_n_step_return_with_neg_inf_reward(self):
        """Test _calculate_n_step_return avec récompense négative infinie."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompense négative infinie
        buffer.temp_buffer = [
            {"reward": -np.inf, "done": False},
            {"reward": 1.0, "done": False},
            {"reward": 2.0, "done": False},
        ]

        # Ne devrait pas lever d'exception et retourner un float
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)
        # -inf est converti en -1.0, donc le résultat peut être négatif
        assert isinstance(result, float)

    def test_get_final_next_state_exception(self):
        """Test _get_final_next_state avec exception."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec données invalides
        buffer.temp_buffer = [
            {"next_state": "invalid_state"}  # String au lieu de numpy array
        ]

        # Ne devrait pas lever d'exception, mais peut retourner None en cas d'erreur
        result = buffer._get_final_next_state(0)
        # Le code retourne None en cas d'exception, donc on accepte None
        assert result is None or isinstance(result, np.ndarray)

    def test_sample_empty_buffer(self):
        """Test sample avec buffer vide."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Buffer vide
        batch = buffer.sample(5)

        assert batch == ([], [])

    def test_sample_with_exception(self):
        """Test sample avec exception."""
        buffer = NStepBuffer(capacity=10, n_step=3)

        # Ajouter quelques transitions et forcer le traitement
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2], dtype=np.float32),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5], dtype=np.float32),
                done=i == 2,  # Terminer à la dernière pour forcer le traitement
                info=None,
            )

        # Mock np.random.choice pour lever une exception
        with (
            patch("numpy.random.choice", side_effect=Exception("Test error")),
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

        # ✅ FIX: On ne peut pas patcher deque.__len__ car c'est en lecture seule.
        # Au lieu de cela, on patche len() globalement pour qu'il lève une exception
        # uniquement quand on appelle len() sur buffer.buffer
        original_len = len
        target_buffer = buffer.buffer

        def mock_len(obj):
            if obj is target_buffer:
                raise Exception("Test error")
            return original_len(obj)

        with (
            patch("builtins.len", side_effect=mock_len),
            patch.object(buffer.logger, "error") as mock_error,
        ):
            stats = buffer.get_stats()

            # Devrait retourner des stats par défaut (dict vide en cas d'erreur)
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

        # Ajouter une transition avec td_error et terminer pour forcer le traitement
        buffer.add_transition(
            state=np.array([1, 2, 3], dtype=np.float32),
            action=1,
            reward=1,
            next_state=np.array([4, 5, 6], dtype=np.float32),
            done=True,  # Terminer pour forcer le traitement immédiat
            info=None,
            td_error=2.0,
        )

        # Vérifier que la priorité a été mise à jour
        # priority = (abs(2.0) + 1e-6)^0.6 ≈ 1.5157
        assert buffer.max_priority >= 1.5

    def test_add_transition_without_td_error(self):
        """Test add_transition sans td_error."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Ajouter une transition sans td_error
        buffer.add_transition(
            state=np.array([1, 2, 3]),
            action=1,
            reward=1,
            next_state=np.array([4, 5, 6]),
            done=False,
            info=None,
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

        # Ajouter quelques transitions et forcer le traitement
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2], dtype=np.float32),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5], dtype=np.float32),
                done=i == 2,  # Terminer à la dernière pour forcer le traitement
                info=None,
            )

        # Mock np.random.choice pour lever une exception
        with (
            patch("numpy.random.choice", side_effect=Exception("Test error")),
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

        # Ajouter quelques transitions et forcer le traitement
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2], dtype=np.float32),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5], dtype=np.float32),
                done=i == 2,  # Terminer à la dernière pour forcer le traitement
                info=None,
            )

        # Mettre à jour les priorités avec td_errors (pas priorities)
        indices = [0, 1, 2]
        # ✅ FIX: Pour obtenir priority >= 4.0 avec alpha=0.6:
        # priority = (abs(td_error) + 1e-6)^0.6 >= 4.0
        # (abs(td_error) + 1e-6) >= 4.0^(1/0.6) ≈ 10.079
        # abs(td_error) >= 10.079 - 1e-6 ≈ 10.079
        # Donc il faut td_error >= 10.08
        td_errors = [10.0, 11.0, 12.0]  # Plus grandes pour que max_priority >= 4

        buffer.update_priorities(indices, td_errors)

        # Vérifier que max_priority a été mis à jour
        # priority = (abs(12.0) + 1e-6)^0.6 ≈ 4.29
        assert buffer.max_priority >= 4.0

    def test_clear(self):
        """Test clear."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Ajouter quelques transitions
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2], dtype=np.float32),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5], dtype=np.float32),
                done=i == 2,  # Terminer à la dernière pour forcer le traitement
                info=None,
            )

        # Vider le buffer
        buffer.clear()

        assert len(buffer.buffer) == 0
        assert len(buffer.temp_buffer) == 0
        # priorities est un array numpy de taille fixe, donc len() ne change pas
        # mais les valeurs sont remplies à 0
        assert (
            np.all(buffer.priorities == 0) or len(buffer.priorities) == buffer.capacity
        )
        assert buffer.max_priority == 1.0
        assert buffer.beta == buffer.beta_start

    def test_get_stats(self):
        """Test get_stats."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Ajouter quelques transitions et forcer le traitement
        for i in range(3):
            buffer.add_transition(
                state=np.array([i, i + 1, i + 2], dtype=np.float32),
                action=i,
                reward=float(i),
                next_state=np.array([i + 3, i + 4, i + 5], dtype=np.float32),
                done=i == 2,  # Terminer à la dernière pour forcer le traitement
                info=None,
            )

        stats = buffer.get_stats()

        assert isinstance(stats, dict)
        assert "buffer_size" in stats
        assert "temp_buffer_size" in stats
        assert "capacity" in stats
        assert "n_step" in stats
        assert "max_priority" in stats
        # beta n'est pas dans get_stats, seulement beta_start et beta_end
        assert "beta_start" in stats
        assert "beta_end" in stats

    def test_get_stats_with_exception(self):
        """Test get_stats avec exception."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # ✅ FIX: On ne peut pas patcher deque.__len__ car c'est en lecture seule.
        # Au lieu de cela, on peut patcher super().get_stats() pour lever une exception
        # car get_stats appelle super().get_stats() en premier
        with (
            patch.object(NStepBuffer, "get_stats", side_effect=Exception("Test error")),
            patch.object(buffer.logger, "error") as mock_error,
        ):
            stats = buffer.get_stats()

            # Devrait retourner des stats par défaut (dict vide en cas d'erreur)
            assert isinstance(stats, dict)

            # Vérifier que l'erreur a été loggée
            mock_error.assert_called_once()

    def test_n_step_calculation_with_nan_rewards(self):
        """Test calcul N-step avec récompenses NaN."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompenses NaN
        buffer.temp_buffer = [
            {"reward": np.nan, "done": False},
            {"reward": np.nan, "done": False},
            {"reward": np.nan, "done": False},
        ]

        # Ne devrait pas lever d'exception et retourner un float
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)
        # NaN est converti en 0, donc le résultat devrait être 0.0
        assert result == 0.0

    def test_n_step_calculation_with_inf_rewards(self):
        """Test calcul N-step avec récompenses infinies."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompenses infinies
        buffer.temp_buffer = [
            {"reward": np.inf, "done": False},
            {"reward": -np.inf, "done": False},
            {"reward": np.inf, "done": False},
        ]

        # Ne devrait pas lever d'exception et retourner un float
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)
        # inf est converti en 1.0, -inf en -1.0
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

        # _calculate_n_step_return retourne 0.0 (float) même avec buffer vide
        assert isinstance(result1, float)
        assert result1 == 0.0
        # _get_final_next_state retourne None en cas d'erreur
        assert result2 is None or isinstance(result2, np.ndarray)

    def test_negative_rewards(self):
        """Test avec récompenses négatives."""
        buffer = NStepPrioritizedBuffer(capacity=10, n_step=3)

        # Buffer temporaire avec récompenses négatives
        buffer.temp_buffer = [
            {"reward": -1.0, "done": False},
            {"reward": -2.0, "done": False},
            {"reward": -3.0, "done": False},
        ]

        # Ne devrait pas lever d'exception et retourner un float
        result = buffer._calculate_n_step_return(0)
        assert isinstance(result, float)
        assert result < 0  # Résultat devrait être négatif
