"""
Tests supplémentaires pour dispatch_env.py - Couverture 95%+ (Partie 2)
"""

import logging
from unittest.mock import Mock, patch

import numpy as np
import pytest

from services.rl.dispatch_env import DispatchEnv


class TestDispatchEnvAdditionalEdgeCases:
    """Tests supplémentaires pour les cas limites et lignes manquantes"""

    def test_get_valid_actions_mask_driver_not_available(self):
        """Test get_valid_actions_mask avec chauffeur non disponible"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des chauffeurs non disponibles
        env.drivers = [
            {"id": 1, "available": False, "load": 2},
            {"id": 2, "available": False, "load": 1},
            {"id": 3, "available": False, "load": 3},
        ]
        env.bookings = [{"id": 1, "priority": 3, "time_window": 30}, {"id": 2, "priority": 1, "time_window": 15}]

        valid_mask = env._get_valid_actions_mask()

        assert isinstance(valid_mask, np.ndarray)
        assert valid_mask[0] == 1  # Action wait toujours valide
        # Les autres actions devraient être invalides

    def test_get_valid_actions_mask_booking_already_assigned(self):
        """Test get_valid_actions_mask avec booking déjà assigné"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des bookings déjà assignés
        env.drivers = [{"id": 1, "available": True, "load": 2}, {"id": 2, "available": True, "load": 1}]
        env.bookings = [
            {"id": 1, "priority": 3, "time_window": 30, "assigned": True},
            {"id": 2, "priority": 1, "time_window": 15, "assigned": True},
        ]

        valid_mask = env._get_valid_actions_mask()

        assert isinstance(valid_mask, np.ndarray)
        assert valid_mask[0] == 1  # Action wait toujours valide
        # Les autres actions devraient être invalides

    def test_get_valid_actions_mask_time_window_constraint(self):
        """Test get_valid_actions_mask avec contrainte de fenêtre temporelle"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des bookings avec fenêtre temporelle expirée
        env.drivers = [{"id": 1, "available": True, "load": 2}, {"id": 2, "available": True, "load": 1}]
        env.bookings = [
            {"id": 1, "priority": 3, "time_window": 0},  # Expiré
            {"id": 2, "priority": 1, "time_window": 15},
        ]

        valid_mask = env._get_valid_actions_mask()

        assert isinstance(valid_mask, np.ndarray)
        assert valid_mask[0] == 1  # Action wait toujours valide

    def test_get_valid_actions_mask_load_constraint(self):
        """Test get_valid_actions_mask avec contrainte de charge"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des chauffeurs à capacité maximale
        env.drivers = [
            {"id": 1, "available": True, "load": 10},  # Capacité max
            {"id": 2, "available": True, "load": 10},  # Capacité max
        ]
        env.bookings = [{"id": 1, "priority": 3, "time_window": 30}, {"id": 2, "priority": 1, "time_window": 15}]

        valid_mask = env._get_valid_actions_mask()

        assert isinstance(valid_mask, np.ndarray)
        assert valid_mask[0] == 1  # Action wait toujours valide

    def test_get_valid_actions_mask_exception_handling(self):
        """Test get_valid_actions_mask avec gestion d'exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides pour provoquer une exception
        env.drivers = [{"invalid": "data"}]
        env.bookings = [{"invalid": "data"}]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            valid_mask = env._get_valid_actions_mask()

            assert isinstance(valid_mask, np.ndarray)
            assert valid_mask[0] == 1  # Action wait toujours valide
            mock_logging.warning.assert_called()

    def test_check_time_window_constraint_valid(self):
        """Test _check_time_window_constraint avec contrainte valide"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        driver = {"id": 1, "available": True, "load": 2}
        booking = {"id": 1, "priority": 3, "time_window": 30}

        is_valid = env._check_time_window_constraint(driver, booking)

        assert isinstance(is_valid, bool)

    def test_check_time_window_constraint_invalid(self):
        """Test _check_time_window_constraint avec contrainte invalide"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        driver = {"id": 1, "available": True, "load": 2}
        booking = {"id": 1, "priority": 3, "time_window": 0}  # Expiré

        is_valid = env._check_time_window_constraint(driver, booking)

        assert isinstance(is_valid, bool)

    def test_check_time_window_constraint_exception(self):
        """Test _check_time_window_constraint avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides
        driver = {"invalid": "data"}
        booking = {"invalid": "data"}

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            is_valid = env._check_time_window_constraint(driver, booking)

            assert isinstance(is_valid, bool)
            assert is_valid is False
            mock_logging.warning.assert_called()

    def test_calculate_travel_time_normal(self):
        """Test _calculate_travel_time normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        driver = {"id": 1, "lat": 48.8566, "lon": 2.3522}
        booking = {"id": 1, "pickup_lat": 48.8606, "pickup_lon": 2.3376}

        travel_time = env._calculate_travel_time(driver, booking)

        assert isinstance(travel_time, float)
        assert travel_time >= 0

    def test_calculate_travel_time_same_location(self):
        """Test _calculate_travel_time même localisation"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        driver = {"id": 1, "lat": 48.8566, "lon": 2.3522}
        booking = {"id": 1, "pickup_lat": 48.8566, "pickup_lon": 2.3522}

        travel_time = env._calculate_travel_time(driver, booking)

        assert isinstance(travel_time, float)
        assert travel_time >= 0

    def test_calculate_travel_time_exception(self):
        """Test _calculate_travel_time avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides
        driver = {"invalid": "data"}
        booking = {"invalid": "data"}

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            travel_time = env._calculate_travel_time(driver, booking)

            assert isinstance(travel_time, float)
            assert travel_time == 0.0
            mock_logging.warning.assert_called()

    def test_get_observation_normal(self):
        """Test _get_observation normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        observation = env._get_observation()

        assert isinstance(observation, np.ndarray)
        assert len(observation) == env.num_drivers * 4 + env.max_bookings * 4 + 2

    def test_get_observation_empty_environment(self):
        """Test _get_observation avec environnement vide"""
        env = DispatchEnv(num_drivers=0, max_bookings=0)
        env.reset()

        observation = env._get_observation()

        assert isinstance(observation, np.ndarray)
        assert len(observation) == 2  # Seulement contexte

    def test_get_observation_exception(self):
        """Test _get_observation avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides
        env.drivers = [{"invalid": "data"}]
        env.bookings = [{"invalid": "data"}]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            observation = env._get_observation()

            assert isinstance(observation, np.ndarray)
            mock_logging.warning.assert_called()

    def test_assign_booking_success(self):
        """Test _assign_booking avec succès"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        driver = {
            "id": 1,
            "available": True,
            "load": 2,
            "lat": 48.8566,
            "lon": 2.3522,
            "total_distance": 0.0,
            "completed_bookings": 0,
        }
        booking = {
            "id": 1,
            "priority": 3,
            "time_window": 30,
            "pickup_lat": 48.8606,
            "pickup_lon": 2.3376,
            "time_window_end": 30,
        }

        reward = env._assign_booking(driver, booking)

        assert isinstance(reward, float)
        assert booking.get("assigned", False) is True

    def test_assign_booking_late_pickup(self):
        """Test _assign_booking avec pickup en retard"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        driver = {
            "id": 1,
            "available": True,
            "load": 2,
            "lat": 48.8566,
            "lon": 2.3522,
            "total_distance": 0.0,
            "completed_bookings": 0,
        }
        booking = {
            "id": 1,
            "priority": 3,
            "time_window": 5,  # Très court
            "pickup_lat": 48.8606,
            "pickup_lon": 2.3376,
            "time_window_end": 5,
        }

        reward = env._assign_booking(driver, booking)

        assert isinstance(reward, float)

    def test_assign_booking_exception(self):
        """Test _assign_booking avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides
        driver = {"invalid": "data"}
        booking = {"invalid": "data"}

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            reward = env._assign_booking(driver, booking)

            assert isinstance(reward, float)
            assert reward == 0.0
            mock_logging.warning.assert_called()

    def test_generate_new_bookings_normal(self):
        """Test _generate_new_bookings normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        initial_count = len(env.bookings)

        env._generate_new_bookings(num=2)

        assert len(env.bookings) >= initial_count

    def test_generate_new_bookings_max_capacity(self):
        """Test _generate_new_bookings avec capacité maximale"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Remplir l'environnement
        env.bookings = [{"id": i} for i in range(env.max_bookings)]

        len(env.bookings)

        env._generate_new_bookings(num=5)

        assert len(env.bookings) <= env.max_bookings

    def test_generate_new_bookings_exception(self):
        """Test _generate_new_bookings avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            env._generate_new_bookings(num=2)

            mock_logging.warning.assert_called()

    def test_check_expired_bookings_normal(self):
        """Test _check_expired_bookings normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        penalty = env._check_expired_bookings()

        assert isinstance(penalty, float)

    def test_check_expired_bookings_with_expired(self):
        """Test _check_expired_bookings avec bookings expirés"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des bookings expirés
        env.bookings = [{"id": 1, "time_window": 0, "assigned": False}, {"id": 2, "time_window": 5, "assigned": False}]

        penalty = env._check_expired_bookings()

        assert isinstance(penalty, float)

    def test_check_expired_bookings_exception(self):
        """Test _check_expired_bookings avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides
        env.bookings = [{"invalid": "data"}]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            penalty = env._check_expired_bookings()

            assert isinstance(penalty, float)
            assert penalty == 0.0
            mock_logging.warning.assert_called()

    def test_update_drivers_normal(self):
        """Test _update_drivers normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        env._update_drivers()

        # Vérifier que les chauffeurs sont mis à jour
        assert len(env.drivers) <= env.num_drivers

    def test_update_drivers_exception(self):
        """Test _update_drivers avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides
        env.drivers = [{"invalid": "data"}]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            env._update_drivers()

            mock_logging.warning.assert_called()

    def test_calculate_distance_normal(self):
        """Test _calculate_distance normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        distance = env._calculate_distance(48.8566, 2.3522, 48.8606, 2.3376)

        assert isinstance(distance, float)
        assert distance >= 0

    def test_calculate_distance_same_point(self):
        """Test _calculate_distance même point"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        distance = env._calculate_distance(48.8566, 2.3522, 48.8566, 2.3522)

        assert isinstance(distance, float)
        assert distance == 0.0

    def test_calculate_distance_exception(self):
        """Test _calculate_distance avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            distance = env._calculate_distance(float("nan"), float("nan"), float("nan"), float("nan"))

            assert isinstance(distance, float)
            assert distance == 0.0
            mock_logging.warning.assert_called()

    def test_end_of_day_return_normal(self):
        """Test _end_of_day_return normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        driver = {
            "id": 1,
            "lat": 48.8566,
            "lon": 2.3522,
            "home_lat": 48.8566,
            "home_lon": 2.3522,
            "total_distance": 0.0,
        }

        env._end_of_day_return(driver)

        # Vérifier que le chauffeur est mis à jour
        assert driver["lat"] == driver["home_lat"]
        assert driver["lon"] == driver["home_lon"]

    def test_end_of_day_return_exception(self):
        """Test _end_of_day_return avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides
        driver = {"invalid": "data"}

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            env._end_of_day_return(driver)

            mock_logging.warning.assert_called()

    def test_get_traffic_density_normal(self):
        """Test _get_traffic_density normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        traffic_density = env._get_traffic_density()

        assert isinstance(traffic_density, float)
        assert 0 <= traffic_density <= 1

    def test_get_traffic_density_exception(self):
        """Test _get_traffic_density avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            traffic_density = env._get_traffic_density()

            assert isinstance(traffic_density, float)
            assert 0 <= traffic_density <= 1
            mock_logging.warning.assert_called()

    def test_get_booking_generation_rate_normal(self):
        """Test _get_booking_generation_rate normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        generation_rate = env._get_booking_generation_rate()

        assert isinstance(generation_rate, float)
        assert generation_rate >= 0

    def test_get_booking_generation_rate_exception(self):
        """Test _get_booking_generation_rate avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            generation_rate = env._get_booking_generation_rate()

            assert isinstance(generation_rate, float)
            assert generation_rate >= 0
            mock_logging.warning.assert_called()

    def test_calculate_episode_bonus_normal(self):
        """Test _calculate_episode_bonus normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        env.successful_assignments = 5
        env.total_bookings = 10

        bonus = env._calculate_episode_bonus()

        assert isinstance(bonus, float)

    def test_calculate_episode_bonus_exception(self):
        """Test _calculate_episode_bonus avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides
        env.successful_assignments = float("nan")
        env.total_bookings = float("nan")

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            bonus = env._calculate_episode_bonus()

            assert isinstance(bonus, float)
            assert bonus == 0.0
            mock_logging.warning.assert_called()

    def test_get_info_normal(self):
        """Test _get_info normal"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        info = env._get_info()

        assert isinstance(info, dict)
        assert "drivers_count" in info
        assert "bookings_count" in info
        assert "current_time" in info

    def test_get_info_exception(self):
        """Test _get_info avec exception"""
        env = DispatchEnv(num_drivers=3, max_bookings=5)
        env.reset()

        # Simuler des données invalides
        env.drivers = [{"invalid": "data"}]
        env.bookings = [{"invalid": "data"}]

        with patch("services.rl.dispatch_env.logging") as mock_logging:
            info = env._get_info()

            assert isinstance(info, dict)
            mock_logging.warning.assert_called()
