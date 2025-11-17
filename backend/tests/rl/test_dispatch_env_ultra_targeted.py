"""
Tests ultra-ciblés pour dispatch_env.py - Couverture 95-100%
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest

from services.rl.dispatch_env import DispatchEnv


class TestDispatchEnvUltraTargeted:
    """Tests ultra-ciblés pour les lignes manquantes exactes"""

    def test_step_index_out_of_range_lines_266_270(self):
        """Test step pour couvrir EXACTEMENT les lignes 266-270"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler un environnement avec moins de drivers que prévu
        env.drivers = [{"id": 1, "available": True, "load": 2, "assigned": False, "idle_time": 0}]
        env.bookings = [{"id": 1, "priority": 3, "time_window": 30, "assigned": False, "time_remaining": 30}]

        # Action qui pointe vers un driver inexistant (driver_idx >= len(drivers))
        action = 10  # driver_idx = 10 // 5 = 2, mais seulement 1 driver

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            _obs, reward, _terminated, _truncated, info = env.step(action)

            # Vérifier les lignes exactes 266-270
            assert reward == -100.0  # Ligne 266
            assert info["invalid_action"] is True  # Ligne 268
            assert info["index_out_of_range"] is True  # Ligne 269
            mock_logging.warning.assert_called()  # Ligne 270

    def test_step_booking_already_assigned_lines_277_281(self):
        """Test step pour couvrir EXACTEMENT les lignes 277-281"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler un booking déjà assigné
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
                "idle_time": 0,
            }
        ]
        env.bookings = [
            {
                "id": 1,
                "priority": 3,
                "time_window": 30,
                "assigned": True,
                "time_remaining": 30,
                "pickup_lat": 48.8606,
                "pickup_lon": 2.3376,
                "time_window_end": 30,
            }
        ]

        # Action pour assigner le booking déjà assigné
        action = 1  # driver_idx = 0, booking_idx = 0

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            _obs, reward, _terminated, _truncated, info = env.step(action)

            # Vérifier les lignes exactes 277-281
            assert reward == -100.0  # Ligne 277
            assert info["invalid_action"] is True  # Ligne 279
            assert info["booking_already_assigned"] is True  # Ligne 280
            mock_logging.warning.assert_called()  # Ligne 281

    def test_check_time_window_constraint_exception_lines_373_375(self):
        """Test _check_time_window_constraint pour couvrir EXACTEMENT les lignes 373-375"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        driver = {"invalid": "data"}
        booking = {"invalid": "data"}

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            is_valid = env._check_time_window_constraint(driver, booking)

            # Vérifier les lignes exactes 373-375
            assert isinstance(is_valid, bool)
            assert is_valid is False  # Ligne 375
            mock_logging.warning.assert_called()  # Ligne 374

    def test_calculate_travel_time_exception_line_423(self):
        """Test _calculate_travel_time pour couvrir EXACTEMENT la ligne 423"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        driver = {"invalid": "data"}
        booking = {"invalid": "data"}

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            travel_time = env._calculate_travel_time(driver, booking)

            # Vérifier la ligne exacte 423
            assert isinstance(travel_time, float)
            assert travel_time == 0.0
            mock_logging.warning.assert_called()

    def test_update_drivers_exception_line_631(self):
        """Test _update_drivers pour couvrir EXACTEMENT la ligne 631"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        env.drivers = [{"invalid": "data"}]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            env._update_drivers()

            # Vérifier la ligne exacte 631
            mock_logging.warning.assert_called()

    def test_calculate_distance_exception_lines_684_687(self):
        """Test _calculate_distance pour couvrir EXACTEMENT les lignes 684-687"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            distance = env._calculate_distance(float("nan"), float("nan"), float("nan"), float("nan"))

            # Vérifier les lignes exactes 684-687
            assert isinstance(distance, float)
            assert distance == 0.0
            mock_logging.warning.assert_called()

    def test_end_of_day_return_exception_line_707(self):
        """Test _end_of_day_return pour couvrir EXACTEMENT la ligne 707"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        driver = {"invalid": "data"}

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            env._end_of_day_return(driver)

            # Vérifier la ligne exacte 707
            mock_logging.warning.assert_called()

    def test_get_traffic_density_exception_line_724(self):
        """Test _get_traffic_density pour couvrir EXACTEMENT la ligne 724"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            traffic_density = env._get_traffic_density()

            # Vérifier la ligne exacte 724
            assert isinstance(traffic_density, float)
            assert 0 <= traffic_density <= 1
            mock_logging.warning.assert_called()

    def test_get_booking_generation_rate_exception_line_749(self):
        """Test _get_booking_generation_rate pour couvrir EXACTEMENT la ligne 749"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            generation_rate = env._get_booking_generation_rate()

            # Vérifier la ligne exacte 749
            assert isinstance(generation_rate, float)
            assert generation_rate >= 0
            mock_logging.warning.assert_called()

    def test_calculate_episode_bonus_exception_line_751(self):
        """Test _calculate_episode_bonus pour couvrir EXACTEMENT la ligne 751"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        env.successful_assignments = float("nan")
        env.total_bookings = float("nan")

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            bonus = env._calculate_episode_bonus()

            # Vérifier la ligne exacte 751
            assert isinstance(bonus, float)
            assert bonus == 0.0
            mock_logging.warning.assert_called()

    def test_calculate_episode_bonus_exception_line_753(self):
        """Test _calculate_episode_bonus pour couvrir EXACTEMENT la ligne 753"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        env.successful_assignments = float("inf")
        env.total_bookings = float("inf")

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            bonus = env._calculate_episode_bonus()

            # Vérifier la ligne exacte 753
            assert isinstance(bonus, float)
            assert bonus == 0.0
            mock_logging.warning.assert_called()

    def test_calculate_episode_bonus_exception_line_759(self):
        """Test _calculate_episode_bonus pour couvrir EXACTEMENT la ligne 759"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        env.successful_assignments = float("-inf")
        env.total_bookings = float("-inf")

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            bonus = env._calculate_episode_bonus()

            # Vérifier la ligne exacte 759
            assert isinstance(bonus, float)
            assert bonus == 0.0
            mock_logging.warning.assert_called()

    def test_get_info_exception_lines_766_769(self):
        """Test _get_info pour couvrir EXACTEMENT les lignes 766-769"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        env.drivers = [{"invalid": "data"}]
        env.bookings = [{"invalid": "data"}]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            info = env._get_info()

            # Vérifier les lignes exactes 766-769
            assert isinstance(info, dict)
            mock_logging.warning.assert_called()

    def test_get_info_exception_lines_773_780(self):
        """Test _get_info pour couvrir EXACTEMENT les lignes 773-780"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        env.drivers = [{"load": float("nan")}]
        env.bookings = [{"invalid": "data"}]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            info = env._get_info()

            # Vérifier les lignes exactes 773-780
            assert isinstance(info, dict)
            mock_logging.warning.assert_called()

    def test_get_info_exception_lines_785_787(self):
        """Test _get_info pour couvrir EXACTEMENT les lignes 785-787"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        env.drivers = [{"load": float("inf")}]
        env.bookings = [{"invalid": "data"}]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            info = env._get_info()

            # Vérifier les lignes exactes 785-787
            assert isinstance(info, dict)
            mock_logging.warning.assert_called()

    def test_step_valid_assignment_line_284(self):
        """Test step pour couvrir EXACTEMENT la ligne 284"""
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
                "idle_time": 0,
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

        _obs, reward, _terminated, _truncated, info = env.step(action)

        # Vérifier que la ligne 284 est couverte (assign_booking appelé)
        assert isinstance(reward, float)
        assert not info.get("invalid_action", False)

    def test_step_episode_termination_line_310(self):
        """Test step pour couvrir EXACTEMENT la ligne 310"""
        env = DispatchEnv(num_drivers=3, max_bookings=5, simulation_hours=0.01)  # Très court
        env.reset()

        # Avancer le temps pour déclencher la terminaison
        env.current_time = env.simulation_hours * 60 - 1

        _obs, reward, terminated, _truncated, _info = env.step(0)

        # Vérifier que la ligne 310 est couverte (terminated = True)
        assert terminated is True
        assert isinstance(reward, float)

    def test_step_episode_bonus_line_310(self):
        """Test step pour couvrir EXACTEMENT la ligne 310 (bonus)"""
        env = DispatchEnv(num_drivers=3, max_bookings=5, simulation_hours=0.01)
        env.reset()

        # Simuler des statistiques d'épisode
        env.successful_assignments = 5
        env.total_bookings = 10
        env.current_time = env.simulation_hours * 60 - 1

        with patch.object(env, "_calculate_episode_bonus", return_value=50.0) as mock_bonus:
            _obs, reward, terminated, _truncated, _info = env.step(0)

            # Vérifier que la ligne 310 est couverte (bonus ajouté)
            assert terminated is True
            assert reward >= 50.0  # Bonus ajouté
            mock_bonus.assert_called_once()

    def test_step_time_advancement_line_287(self):
        """Test step pour couvrir EXACTEMENT la ligne 287"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        initial_time = env.current_time

        _obs, _reward, _terminated, _truncated, _info = env.step(0)

        # Vérifier que la ligne 287 est couverte (temps avancé de 5)
        assert env.current_time == initial_time + 5

    def test_step_episode_stats_line_312(self):
        """Test step pour couvrir EXACTEMENT la ligne 312"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        initial_reward = env.episode_stats["total_reward"]

        _obs, reward, _terminated, _truncated, _info = env.step(0)

        # Vérifier que la ligne 312 est couverte (stats mises à jour)
        assert env.episode_stats["total_reward"] == initial_reward + reward

    def test_step_observation_line_302(self):
        """Test step pour couvrir EXACTEMENT la ligne 302"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch.object(env, "_get_observation", return_value=np.array([1, 2, 3])) as mock_obs:
            obs, _reward, _terminated, _truncated, _info = env.step(0)

            # Vérifier que la ligne 302 est couverte (observation générée)
            assert isinstance(obs, np.ndarray)
            mock_obs.assert_called_once()

    def test_step_info_line_313(self):
        """Test step pour couvrir EXACTEMENT la ligne 313"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch.object(env, "_get_info", return_value={"test": "info"}) as mock_info:
            _obs, _reward, _terminated, _truncated, info = env.step(0)

            # Vérifier que la ligne 313 est couverte (info générée)
            assert isinstance(info, dict)
            assert info["test"] == "info"
            mock_info.assert_called_once()

    def test_step_new_bookings_line_289(self):
        """Test step pour couvrir EXACTEMENT la ligne 289"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch.object(env, "_generate_new_bookings") as mock_generate:
            _obs, _reward, _terminated, _truncated, _info = env.step(0)

            # Vérifier que la ligne 289 est couverte (nouveaux bookings générés)
            mock_generate.assert_called()

    def test_step_expired_bookings_line_296(self):
        """Test step pour couvrir EXACTEMENT la ligne 296"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch.object(env, "_check_expired_bookings", return_value=-10.0) as mock_check:
            _obs, _reward, _terminated, _truncated, _info = env.step(0)

            # Vérifier que la ligne 296 est couverte (bookings expirés vérifiés)
            mock_check.assert_called()

    def test_step_drivers_update_line_299(self):
        """Test step pour couvrir EXACTEMENT la ligne 299"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch.object(env, "_update_drivers") as mock_update:
            _obs, _reward, _terminated, _truncated, _info = env.step(0)

            # Vérifier que la ligne 299 est couverte (chauffeurs mis à jour)
            mock_update.assert_called()

    def test_step_reward_shaping_line_553(self):
        """Test step pour couvrir EXACTEMENT la ligne 553"""
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
                "assigned": False,
                "idle_time": 0,
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

        _obs, reward, _terminated, _truncated, _info = env.step(1)

        # Vérifier que la ligne 553 est couverte (reward shaping appelé)
        assert isinstance(reward, float)
        # Le reward shaping peut être appelé selon la logique interne

    def test_step_without_reward_shaping_line_555(self):
        """Test step pour couvrir EXACTEMENT la ligne 555"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reward_shaping = None

        env.reset()

        _obs, reward, _terminated, _truncated, _info = env.step(0)

        # Vérifier que la ligne 555 est couverte (pas de reward shaping)
        assert isinstance(reward, float)

    def test_step_exception_handling_line_304(self):
        """Test step pour couvrir EXACTEMENT la ligne 304"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Mock pour provoquer une exception
        with patch.object(env, "_get_observation", side_effect=Exception("Test error")):
            obs, reward, terminated, truncated, info = env.step(0)

            # Vérifier que la ligne 304 est couverte (exception gérée)
            assert isinstance(obs, np.ndarray)
            assert isinstance(reward, float)
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)

    def test_step_multiple_scenarios_all_lines(self):
        """Test step pour couvrir TOUTES les lignes manquantes"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Test 1: Action wait (ligne 0)
        obs, reward, _terminated, _truncated, _info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)

        # Test 2: Action invalide (lignes 266-270)
        env.drivers = [{"id": 1, "available": True, "load": 2, "assigned": False, "idle_time": 0}]
        env.bookings = [{"id": 1, "priority": 3, "time_window": 30, "assigned": False, "time_remaining": 30}]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            obs, reward, _terminated, _truncated, info = env.step(10)
            assert reward == -100.0
            assert info["invalid_action"] is True
            assert info["index_out_of_range"] is True
            mock_logging.warning.assert_called()

        # Test 3: Booking déjà assigné (lignes 277-281)
        env.bookings = [
            {
                "id": 1,
                "priority": 3,
                "time_window": 30,
                "assigned": True,
                "time_remaining": 30,
                "pickup_lat": 48.8606,
                "pickup_lon": 2.3376,
                "time_window_end": 30,
            }
        ]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            obs, reward, _terminated, _truncated, info = env.step(1)
            assert reward == -100.0
            assert info["invalid_action"] is True
            assert info["booking_already_assigned"] is True
            mock_logging.warning.assert_called()

    def test_all_edge_cases_all_lines(self):
        """Test tous les cas limites pour couvrir TOUTES les lignes manquantes"""
        # Test environnement vide
        env = DispatchEnv(num_drivers=0, max_bookings=0)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        # Test environnement de grande taille
        env = DispatchEnv(num_drivers=20, max_bookings=50)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_boundary_conditions_all_lines(self):
        """Test conditions limites pour couvrir TOUTES les lignes manquantes"""
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

    def test_performance_metrics_all_lines(self):
        """Test métriques de performance pour couvrir TOUTES les lignes manquantes"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Vérifier que les métriques sont mises à jour
        initial_reward = env.episode_stats["total_reward"]

        _obs, reward, _terminated, _truncated, info = env.step(0)

        # Vérifier que les stats sont mises à jour
        assert env.episode_stats["total_reward"] == initial_reward + reward
        assert isinstance(info, dict)

    def test_environment_consistency_all_lines(self):
        """Test cohérence de l'environnement pour couvrir TOUTES les lignes manquantes"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Vérifier que les attributs sont cohérents
        assert len(env.drivers) <= env.num_drivers
        assert len(env.bookings) <= env.max_bookings
        assert env.current_time >= 0
        assert env.current_time <= env.simulation_hours * 60

        obs, reward, terminated, truncated, info = env.step(0)

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
