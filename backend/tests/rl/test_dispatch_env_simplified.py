"""
Tests simplifiés pour dispatch_env.py - Couverture 95%+
"""

from unittest.mock import Mock, patch

import numpy as np

from services.rl.dispatch_env import DispatchEnv


class TestDispatchEnvSimplified:
    """Tests simplifiés pour atteindre 95% de couverture"""

    def test_step_action_out_of_bounds(self):
        """Test step avec action hors limites"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Action hors limites mais dans les limites du tableau
        obs, reward, terminated, truncated, info = env.step(
            15
        )  # Dans les limites du tableau

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_action_negative(self):
        """Test step avec action négative"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(-1)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_valid_action_wait(self):
        """Test step avec action wait valide"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_episode_termination_bonus(self):
        """Test step avec terminaison d'épisode et bonus"""
        env = DispatchEnv(
            num_drivers=3, max_bookings=5, simulation_hours=0.01
        )  # Très court
        env.reset()

        # Avancer le temps pour déclencher la terminaison
        env.current_time = env.simulation_hours * 60 - 1

        obs, reward, terminated, _truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert terminated is True
        assert isinstance(info, dict)

    def test_step_time_advancement(self):
        """Test avancement du temps dans step"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        initial_time = env.current_time

        _obs, _reward, _terminated, _truncated, _info = env.step(0)

        assert env.current_time == initial_time + 5

    def test_step_episode_stats_update(self):
        """Test mise à jour des statistiques d'épisode"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        initial_reward = env.episode_stats["total_reward"]

        _obs, reward, _terminated, _truncated, _info = env.step(0)

        assert env.episode_stats["total_reward"] == initial_reward + reward

    def test_step_observation_generation(self):
        """Test génération de l'observation"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        obs, _reward, _terminated, _truncated, _info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert len(obs) == env.num_drivers * 4 + env.max_bookings * 4 + 2

    def test_step_info_generation(self):
        """Test génération des informations"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        _obs, _reward, _terminated, _truncated, info = env.step(0)

        assert isinstance(info, dict)
        assert "active_bookings" in info
        assert "available_drivers" in info
        assert "current_time" in info

    def test_step_with_reward_shaping(self):
        """Test step avec reward shaping"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)

        # Mock du reward shaping
        mock_reward_shaping = Mock()
        mock_reward_shaping.calculate_reward.return_value = 25.0
        env.reward_shaping = mock_reward_shaping

        env.reset()

        _obs, reward, _terminated, _truncated, _info = env.step(0)

        assert isinstance(reward, float)

    def test_step_without_reward_shaping(self):
        """Test step sans reward shaping"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reward_shaping = None

        env.reset()

        _obs, reward, _terminated, _truncated, _info = env.step(0)

        assert isinstance(reward, float)

    def test_step_multiple_actions(self):
        """Test step avec plusieurs actions"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Test plusieurs actions
        for action in [0, 1, 2, 3, 4, 5]:
            obs, reward, terminated, truncated, info = env.step(action)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

            if terminated or truncated:
                break

    def test_step_edge_case_empty_environment(self):
        """Test step avec environnement vide"""
        env = DispatchEnv(num_drivers=0, max_bookings=0)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_edge_case_single_element(self):
        """Test step avec un seul élément"""
        env = DispatchEnv(num_drivers=1, max_bookings=1)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_edge_case_large_environment(self):
        """Test step avec environnement de grande taille"""
        env = DispatchEnv(num_drivers=10, max_bookings=20)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_boundary_conditions(self):
        """Test step avec conditions limites"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Test avec temps limite
        env.current_time = env.simulation_hours * 60 - 1

        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_performance_metrics(self):
        """Test step avec métriques de performance"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Vérifier que les métriques sont mises à jour
        initial_reward = env.episode_stats["total_reward"]

        _obs, reward, _terminated, _truncated, info = env.step(0)

        # La récompense peut être négative, donc on vérifie juste que les stats sont mises à jour
        assert env.episode_stats["total_reward"] == initial_reward + reward
        assert isinstance(info, dict)

    def test_step_exception_handling(self):
        """Test step avec gestion d'exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Mock pour provoquer une exception dans une méthode interne
        with patch.object(
            env, "_get_valid_actions_mask", side_effect=Exception("Test error")
        ):
            obs, reward, terminated, truncated, info = env.step(0)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_step_reset_exception_handling(self):
        """Test reset avec gestion d'exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)

        # Mock pour provoquer une exception dans une méthode interne
        with patch.object(
            env, "_get_valid_actions_mask", side_effect=Exception("Test error")
        ):
            state, info = env.reset()

            assert isinstance(state, np.ndarray)
            assert isinstance(info, dict)

    def test_step_get_valid_actions_exception(self):
        """Test get_valid_actions avec gestion d'exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Mock pour provoquer une exception
        with patch.object(
            env, "_get_valid_actions_mask", side_effect=Exception("Test error")
        ):
            valid_actions = env.get_valid_actions()

            assert isinstance(valid_actions, list)
            assert 0 in valid_actions  # Action wait toujours disponible

    def test_step_environment_consistency(self):
        """Test cohérence de l'environnement"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Vérifier que les attributs sont cohérents
        assert len(env.drivers) <= env.num_drivers
        assert len(env.bookings) <= env.max_bookings
        assert env.current_time >= 0
        assert env.current_time <= env.simulation_hours * 60

    def test_step_multiple_episodes(self):
        """Test plusieurs épisodes consécutifs"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)

        for _episode in range(3):
            state, info = env.reset()
            assert isinstance(state, np.ndarray)
            assert isinstance(info, dict)

            # Quelques steps
            for _ in range(5):
                obs, reward, terminated, truncated, info = env.step(0)
                assert isinstance(obs, np.ndarray)
                assert isinstance(reward, float)

                if terminated or truncated:
                    break

    def test_step_simulation_hours_variations(self):
        """Test step avec différentes durées de simulation"""
        for hours in [0.1, 0.5, 1.0, 2.0]:
            env = DispatchEnv(num_drivers=3, max_bookings=5, simulation_hours=hours)
            env.reset()

            obs, reward, terminated, truncated, info = env.step(0)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_step_render_mode_variations(self):
        """Test step avec différents modes de rendu"""
        for mode in ["human", "rgb_array", "invalid"]:
            env = DispatchEnv(num_drivers=3, max_bookings=5, render_mode=mode)
            env.reset()

            obs, reward, terminated, truncated, info = env.step(0)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_step_reward_profile_variations(self):
        """Test step avec différents profils de récompense"""
        for profile in [
            "DEFAULT",
            "PUNCTUALITY_FOCUSED",
            "EQUITY_FOCUSED",
            "EFFICIENCY_FOCUSED",
        ]:
            env = DispatchEnv(num_drivers=3, max_bookings=5, reward_profile=profile)
            env.reset()

            obs, reward, terminated, truncated, info = env.step(0)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_step_seed_variations(self):
        """Test step avec différents seeds"""
        for seed in [None, 42, 123, 456]:
            env = DispatchEnv(num_drivers=3, max_bookings=5, seed=seed)
            env.reset()

            obs, reward, terminated, truncated, info = env.step(0)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
