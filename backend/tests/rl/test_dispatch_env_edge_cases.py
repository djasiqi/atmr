"""
Tests supplémentaires pour dispatch_env.py - Couverture 95%+
"""

from unittest.mock import Mock, patch

import numpy as np

from services.rl.dispatch_env import DispatchEnv


class TestDispatchEnvEdgeCases:
    """Tests pour les cas limites et lignes manquantes"""

    def test_step_index_out_of_range_driver(self):
        """Test step avec index driver hors limites"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler un environnement avec moins de drivers que prévu
        env.drivers = [{"id": 1, "available": True, "load": 2, "assigned": False}]
        env.bookings = [
            {
                "id": 1,
                "priority": 3,
                "time_window": 30,
                "assigned": False,
                "time_remaining": 30,
            }
        ]

        # Action qui pointe vers un driver inexistant (driver_idx >= len(drivers))
        action = 10  # driver_idx = 10 // 5 = 2, mais seulement 1 driver

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            obs, reward, _terminated, _truncated, info = env.step(action)

            assert isinstance(obs, np.ndarray)
            assert reward == -100.0
            assert info["invalid_action"] is True
            assert info["index_out_of_range"] is True
            mock_logging.warning.assert_called()

    def test_step_index_out_of_range_booking(self):
        """Test step avec index booking hors limites"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler un environnement avec moins de bookings que prévu
        env.drivers = [{"id": 1, "available": True, "load": 2, "assigned": False}]
        env.bookings = [
            {
                "id": 1,
                "priority": 3,
                "time_window": 30,
                "assigned": False,
                "time_remaining": 30,
            }
        ]

        # Action qui pointe vers un booking inexistant (booking_idx >= len(bookings))
        action = 3  # driver_idx = 0, booking_idx = 3, mais seulement 1 booking

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            obs, reward, _terminated, _truncated, info = env.step(action)

            assert isinstance(obs, np.ndarray)
            assert reward == -100.0
            assert info["invalid_action"] is True
            assert info["index_out_of_range"] is True
            mock_logging.warning.assert_called()

    def test_step_booking_already_assigned(self):
        """Test step avec booking déjà assigné"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler un booking déjà assigné
        env.drivers = [{"id": 1, "available": True, "load": 2, "assigned": False}]
        env.bookings = [
            {
                "id": 1,
                "priority": 3,
                "time_window": 30,
                "assigned": True,
                "time_remaining": 30,
            }
        ]

        # Action pour assigner le booking déjà assigné
        action = 1  # driver_idx = 0, booking_idx = 0

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            obs, reward, _terminated, _truncated, info = env.step(action)

            assert isinstance(obs, np.ndarray)
            assert reward == -100.0
            assert info["invalid_action"] is True
            assert info["booking_already_assigned"] is True
            mock_logging.warning.assert_called()

    def test_step_valid_assignment(self):
        """Test step avec assignation valide"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler une assignation valide
        env.drivers = [
            {
                "id": 1,
                "available": True,
                "load": 2,
                "lat": 48.8566,
                "lon": 2.3522,
                "total_distance": 0.0,
                "completed_bookings": 0,
                "assigned": False,
            }
        ]
        env.bookings = [
            {
                "id": 1,
                "priority": 3,
                "time_window": 30,
                "pickup_lat": 48.8606,
                "pickup_lon": 2.3376,
                "time_window_end": 30,
                "assigned": False,
                "time_remaining": 30,
            }
        ]

        # Action pour assigner le booking
        action = 1  # driver_idx = 0, booking_idx = 0

        obs, reward, _terminated, _truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert not info.get("invalid_action", False)

    def test_step_episode_termination(self):
        """Test step avec terminaison d'épisode"""
        env = DispatchEnv(
            num_drivers=3, max_bookings=5, simulation_hours=0.1
        )  # 6 minutes
        env.reset()

        # Avancer le temps pour déclencher la terminaison
        env.current_time = 5  # Juste avant la limite

        obs, reward, terminated, truncated, _info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert terminated is True
        assert truncated is False

    def test_step_episode_bonus_calculation(self):
        """Test calcul du bonus d'épisode lors de la terminaison"""
        env = DispatchEnv(num_drivers=3, max_bookings=5, simulation_hours=0.1)
        env.reset()

        # Simuler des statistiques d'épisode
        env.successful_assignments = 5
        env.total_bookings = 10
        env.current_time = 5  # Juste avant la limite

        with patch.object(
            env, "_calculate_episode_bonus", return_value=50.0
        ) as mock_bonus:
            _obs, reward, terminated, _truncated, _info = env.step(0)

            assert terminated is True
            assert reward >= 50.0  # Bonus ajouté
            mock_bonus.assert_called_once()

    def test_step_episode_stats_update(self):
        """Test mise à jour des statistiques d'épisode"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        initial_reward = env.episode_stats["total_reward"]

        _obs, reward, _terminated, _truncated, _info = env.step(0)

        assert env.episode_stats["total_reward"] == initial_reward + reward

    def test_step_time_advancement(self):
        """Test avancement du temps"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        initial_time = env.current_time

        _obs, _reward, _terminated, _truncated, _info = env.step(0)

        assert env.current_time == initial_time + 5

    def test_step_new_bookings_generation(self):
        """Test génération de nouveaux bookings"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        len(env.bookings)

        with patch.object(env, "_generate_new_bookings") as mock_generate:
            _obs, _reward, _terminated, _truncated, _info = env.step(0)

            mock_generate.assert_called()

    def test_step_expired_bookings_check(self):
        """Test vérification des bookings expirés"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch.object(
            env, "_check_expired_bookings", return_value=-10.0
        ) as mock_check:
            _obs, _reward, _terminated, _truncated, _info = env.step(0)

            mock_check.assert_called()

    def test_step_drivers_update(self):
        """Test mise à jour des chauffeurs"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch.object(env, "_update_drivers") as mock_update:
            _obs, _reward, _terminated, _truncated, _info = env.step(0)

            mock_update.assert_called()

    def test_step_observation_generation(self):
        """Test génération de l'observation"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch.object(
            env, "_get_observation", return_value=np.array([1, 2, 3])
        ) as mock_obs:
            obs, _reward, _terminated, _truncated, _info = env.step(0)

            assert isinstance(obs, np.ndarray)
            mock_obs.assert_called()

    def test_step_info_generation(self):
        """Test génération des informations"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch.object(env, "_get_info", return_value={"test": "info"}) as mock_info:
            _obs, _reward, _terminated, _truncated, info = env.step(0)

            assert isinstance(info, dict)
            assert info["test"] == "info"
            mock_info.assert_called()

    def test_step_with_reward_shaping(self):
        """Test step avec reward shaping activé"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)

        # Mock du reward shaping
        mock_reward_shaping = Mock()
        mock_reward_shaping.calculate_reward.return_value = 25.0
        env.reward_shaping = mock_reward_shaping

        env.reset()

        # Simuler une assignation qui utilise le reward shaping
        env.drivers = [
            {
                "id": 1,
                "available": True,
                "load": 2,
                "lat": 48.8566,
                "lon": 2.3522,
                "total_distance": 0.0,
                "completed_bookings": 0,
            }
        ]
        env.bookings = [
            {
                "id": 1,
                "priority": 3,
                "time_window": 30,
                "pickup_lat": 48.8606,
                "pickup_lon": 2.3376,
                "time_window_end": 30,
            }
        ]

        _obs, reward, _terminated, _truncated, _info = env.step(1)

        assert isinstance(reward, float)
        # Le reward shaping peut être appelé selon la logique interne

    def test_step_without_reward_shaping(self):
        """Test step sans reward shaping"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reward_shaping = None

        env.reset()

        _obs, reward, _terminated, _truncated, _info = env.step(0)

        assert isinstance(reward, float)

    def test_step_exception_handling(self):
        """Test gestion d'exception dans step"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Mock pour provoquer une exception
        with patch.object(
            env, "_get_observation", side_effect=Exception("Observation error")
        ):
            obs, reward, terminated, truncated, info = env.step(0)

            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_step_multiple_scenarios(self):
        """Test step avec plusieurs scénarios"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Test 1: Action wait
        obs, reward, _terminated, _truncated, _info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)

        # Test 2: Action invalide
        obs, reward, _terminated, _truncated, _info = env.step(999)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)

        # Test 3: Action négative
        obs, reward, _terminated, _truncated, _info = env.step(-1)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)

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
        env = DispatchEnv(num_drivers=20, max_bookings=50)
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
        initial_stats = env.episode_stats.copy()

        _obs, _reward, _terminated, _truncated, info = env.step(0)

        assert env.episode_stats["total_reward"] >= initial_stats["total_reward"]
        assert isinstance(info, dict)
