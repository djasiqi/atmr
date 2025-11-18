"""
Tests complets pour reward_shaping.py - Couverture 90%+
"""

from unittest.mock import patch

import numpy as np
import pytest

from services.rl.reward_shaping import AdvancedRewardShaping, RewardShapingConfig


class TestRewardShapingComprehensive:
    """Tests complets pour AdvancedRewardShaping"""

    def test_init_with_custom_weights(self):
        """Test initialisation avec poids personnalisés"""
        reward_shaping = AdvancedRewardShaping(
            punctuality_weight=2.0,
            distance_weight=1.0,
            equity_weight=0.5,
            efficiency_weight=0.3,
            satisfaction_weight=0.6,
        )

        assert reward_shaping.punctuality_weight == 2.0
        assert reward_shaping.distance_weight == 1.0
        assert reward_shaping.equity_weight == 0.5
        assert reward_shaping.efficiency_weight == 0.3
        assert reward_shaping.satisfaction_weight == 0.6

    def test_calculate_reward_debug_logging(self):
        """Test calcul de récompense avec logging debug activé"""
        reward_shaping = AdvancedRewardShaping()

        state = np.random.rand(10)
        action = 1
        next_state = np.random.rand(10)
        info = {
            "is_late": False,
            "lateness_minutes": 0,
            "distance_km": 5.0,
            "driver_loads": [1, 2, 3],
            "assignment_successful": True,
            "assignment_time_minutes": 3,
            "driver_type": "REGULAR",
            "respects_preferences": True,
            "booking_priority": 3,
        }

        # Activer le logging debug
        with patch("services.rl.reward_shaping.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True

            reward_shaping.calculate_reward(state, action, next_state, info)

            # Vérifier que le logging debug a été appelé
            mock_logger.debug.assert_called_once()

    def test_punctuality_reward_perfect_punctuality(self):
        """Test récompense ponctualité parfaite"""
        reward_shaping = AdvancedRewardShaping()

        info = {"is_late": False, "lateness_minutes": 0}
        reward = reward_shaping._calculate_punctuality_reward(info)

        assert reward == 100.0

    def test_punctuality_reward_early_arrival(self):
        """Test récompense ponctualité avec arrivée en avance"""
        reward_shaping = AdvancedRewardShaping()

        info = {"is_late": False, "lateness_minutes": 5}
        reward = reward_shaping._calculate_punctuality_reward(info)

        assert reward == 90.0  # 100 - 5*2

    def test_punctuality_reward_outbound_late(self):
        """Test récompense ponctualité ALLER en retard"""
        reward_shaping = AdvancedRewardShaping()

        info = {"is_late": True, "lateness_minutes": 10, "is_outbound": True}
        reward = reward_shaping._calculate_punctuality_reward(info)

        assert reward == -100.0  # -min(200, 10*10)

    def test_punctuality_reward_return_soft_tolerance(self):
        """Test récompense ponctualité RETOUR dans tolérance douce"""
        reward_shaping = AdvancedRewardShaping()

        info = {"is_late": True, "lateness_minutes": 10, "is_outbound": False}
        reward = reward_shaping._calculate_punctuality_reward(info)

        assert reward == 0.0  # Neutre dans la tolérance douce

    def test_punctuality_reward_return_hard_tolerance(self):
        """Test récompense ponctualité RETOUR dans tolérance dure"""
        reward_shaping = AdvancedRewardShaping()

        info = {"is_late": True, "lateness_minutes": 20, "is_outbound": False}
        reward = reward_shaping._calculate_punctuality_reward(info)

        assert reward == -10.0  # -(20-15)*2

    def test_punctuality_reward_return_severe_late(self):
        """Test récompense ponctualité RETOUR très en retard"""
        reward_shaping = AdvancedRewardShaping()

        info = {"is_late": True, "lateness_minutes": 50, "is_outbound": False}
        reward = reward_shaping._calculate_punctuality_reward(info)

        assert reward == -100.0  # -min(100, 50*3)

    def test_distance_reward_short_distance(self):
        """Test récompense distance courte"""
        reward_shaping = AdvancedRewardShaping()

        info = {"distance_km": 3.0}
        reward = reward_shaping._calculate_distance_reward(info)

        assert reward == 28.0  # 20 + (5-3)*4

    def test_distance_reward_long_distance(self):
        """Test récompense distance longue"""
        reward_shaping = AdvancedRewardShaping()

        info = {"distance_km": 20.0}
        reward = reward_shaping._calculate_distance_reward(info)

        # Log penalty: -log(20) * 10 ≈ -30
        assert reward < 0
        assert reward >= -50.0  # Max penalty

    def test_equity_reward_excellent_balance(self):
        """Test récompense équité excellente"""
        reward_shaping = AdvancedRewardShaping()

        info = {"driver_loads": [1, 1, 1]}  # Écart type ≈ 0
        reward = reward_shaping._calculate_equity_reward(info)

        assert reward == 100.0

    def test_equity_reward_good_balance(self):
        """Test récompense équité bonne"""
        reward_shaping = AdvancedRewardShaping()

        info = {"driver_loads": [1, 2, 3]}  # Écart type ≈ 1
        reward = reward_shaping._calculate_equity_reward(info)

        assert reward == 100.0  # < excellent_equity_threshold

    def test_equity_reward_good_balance_threshold(self):
        """Test récompense équité bonne (entre les seuils)"""
        reward_shaping = AdvancedRewardShaping()

        # Créer des charges avec écart type entre 1.0 et 2.0
        info = {"driver_loads": [1, 3, 5]}  # Écart type ≈ 1.63
        reward = reward_shaping._calculate_equity_reward(info)

        assert reward == 50.0  # Bon équilibre

    def test_equity_reward_poor_balance(self):
        """Test récompense équité mauvaise"""
        reward_shaping = AdvancedRewardShaping()

        info = {"driver_loads": [1, 5, 10]}  # Écart type élevé
        reward = reward_shaping._calculate_equity_reward(info)

        assert reward < 0  # Pénalité déséquilibre

    def test_equity_reward_empty_loads(self):
        """Test récompense équité avec charges vides"""
        reward_shaping = AdvancedRewardShaping()

        info = {"driver_loads": []}
        reward = reward_shaping._calculate_equity_reward(info)

        assert reward == 0.0

    def test_equity_reward_single_load(self):
        """Test récompense équité avec une seule charge"""
        reward_shaping = AdvancedRewardShaping()

        info = {"driver_loads": [5]}
        reward = reward_shaping._calculate_equity_reward(info)

        assert reward == 0.0

    def test_efficiency_reward_successful_fast(self):
        """Test récompense efficacité assignation réussie rapide"""
        reward_shaping = AdvancedRewardShaping()

        info = {"assignment_successful": True, "assignment_time_minutes": 3}
        reward = reward_shaping._calculate_efficiency_reward(info)

        assert reward == 70.0  # 50 + 20 (rapide)

    def test_efficiency_reward_successful_medium(self):
        """Test récompense efficacité assignation réussie moyenne"""
        reward_shaping = AdvancedRewardShaping()

        info = {"assignment_successful": True, "assignment_time_minutes": 7}
        reward = reward_shaping._calculate_efficiency_reward(info)

        assert reward == 60.0  # 50 + 10 (moyen)

    def test_efficiency_reward_successful_slow(self):
        """Test récompense efficacité assignation réussie lente"""
        reward_shaping = AdvancedRewardShaping()

        info = {"assignment_successful": True, "assignment_time_minutes": 15}
        reward = reward_shaping._calculate_efficiency_reward(info)

        assert reward == 50.0  # Base seulement

    def test_efficiency_reward_failed(self):
        """Test récompense efficacité assignation échouée"""
        reward_shaping = AdvancedRewardShaping()

        info = {"assignment_successful": False}
        reward = reward_shaping._calculate_efficiency_reward(info)

        assert reward == -20.0

    def test_satisfaction_reward_regular_driver(self):
        """Test récompense satisfaction chauffeur REGULAR"""
        reward_shaping = AdvancedRewardShaping()

        info = {"driver_type": "REGULAR"}
        reward = reward_shaping._calculate_satisfaction_reward(info)

        assert reward == 20.0

    def test_satisfaction_reward_respects_preferences(self):
        """Test récompense satisfaction respect préférences"""
        reward_shaping = AdvancedRewardShaping()

        info = {"respects_preferences": True}
        reward = reward_shaping._calculate_satisfaction_reward(info)

        assert reward == 15.0

    def test_satisfaction_reward_emergency_misuse(self):
        """Test récompense satisfaction mauvaise utilisation EMERGENCY"""
        reward_shaping = AdvancedRewardShaping()

        info = {"driver_type": "EMERGENCY", "booking_priority": 3}
        reward = reward_shaping._calculate_satisfaction_reward(info)

        assert reward == -10.0

    def test_satisfaction_reward_emergency_correct_use(self):
        """Test récompense satisfaction bonne utilisation EMERGENCY"""
        reward_shaping = AdvancedRewardShaping()

        info = {"driver_type": "EMERGENCY", "booking_priority": 4}
        reward = reward_shaping._calculate_satisfaction_reward(info)

        assert reward == 0.0

    def test_satisfaction_reward_default(self):
        """Test récompense satisfaction par défaut"""
        reward_shaping = AdvancedRewardShaping()

        info = {}
        reward = reward_shaping._calculate_satisfaction_reward(info)

        assert reward == 0.0

    def test_update_weights(self):
        """Test mise à jour des poids"""
        reward_shaping = AdvancedRewardShaping()

        with patch("services.rl.reward_shaping.logger") as mock_logger:
            reward_shaping.update_weights(punctuality_weight=2.0, distance_weight=1.5)

            assert reward_shaping.punctuality_weight == 2.0
            assert reward_shaping.distance_weight == 1.5
            assert mock_logger.info.call_count == 2

    def test_update_weights_invalid_attribute(self):
        """Test mise à jour des poids avec attribut invalide"""
        reward_shaping = AdvancedRewardShaping()

        # Ne devrait pas lever d'erreur
        reward_shaping.update_weights(invalid_weight=5.0)

        # Les poids existants ne devraient pas changer
        assert reward_shaping.punctuality_weight == 1.0

    def test_get_current_weights(self):
        """Test récupération des poids actuels"""
        reward_shaping = AdvancedRewardShaping(punctuality_weight=2.0, distance_weight=1.5)

        weights = reward_shaping.get_current_weights()

        assert weights["punctuality_weight"] == 2.0
        assert weights["distance_weight"] == 1.5
        assert weights["equity_weight"] == 0.3
        assert weights["efficiency_weight"] == 0.2
        assert weights["satisfaction_weight"] == 0.4

    def test_reset(self):
        """Test reset des statistiques"""
        reward_shaping = AdvancedRewardShaping()

        with patch("services.rl.reward_shaping.logger") as mock_logger:
            reward_shaping.reset()

            mock_logger.debug.assert_called_once_with("[RewardShaping] Reset des statistiques")


class TestRewardShapingConfigComprehensive:
    """Tests complets pour RewardShapingConfig"""

    def test_get_profile_default(self):
        """Test récupération profil par défaut"""
        profile = RewardShapingConfig.get_profile("DEFAULT")

        assert profile["punctuality_weight"] == 1.0
        assert profile["distance_weight"] == 0.5
        assert profile["equity_weight"] == 0.3
        assert profile["efficiency_weight"] == 0.2
        assert profile["satisfaction_weight"] == 0.4

    def test_get_profile_punctuality_focused(self):
        """Test récupération profil ponctualité"""
        profile = RewardShapingConfig.get_profile("PUNCTUALITY_FOCUSED")

        assert profile["punctuality_weight"] == 1.5
        assert profile["distance_weight"] == 0.3
        assert profile["equity_weight"] == 0.2
        assert profile["efficiency_weight"] == 0.1
        assert profile["satisfaction_weight"] == 0.3

    def test_get_profile_equity_focused(self):
        """Test récupération profil équité"""
        profile = RewardShapingConfig.get_profile("EQUITY_FOCUSED")

        assert profile["punctuality_weight"] == 0.8
        assert profile["distance_weight"] == 0.4
        assert profile["equity_weight"] == 0.6
        assert profile["efficiency_weight"] == 0.2
        assert profile["satisfaction_weight"] == 0.3

    def test_get_profile_efficiency_focused(self):
        """Test récupération profil efficacité"""
        profile = RewardShapingConfig.get_profile("EFFICIENCY_FOCUSED")

        assert profile["punctuality_weight"] == 0.7
        assert profile["distance_weight"] == 1.0
        assert profile["equity_weight"] == 0.2
        assert profile["efficiency_weight"] == 0.4
        assert profile["satisfaction_weight"] == 0.2

    def test_get_profile_invalid(self):
        """Test récupération profil invalide"""
        profile = RewardShapingConfig.get_profile("INVALID_PROFILE")

        # Devrait retourner le profil par défaut
        assert profile == RewardShapingConfig.DEFAULT

    def test_get_profile_case_insensitive(self):
        """Test récupération profil insensible à la casse"""
        profile = RewardShapingConfig.get_profile("punctuality_focused")

        assert profile["punctuality_weight"] == 1.5

    def test_get_weights_default(self):
        """Test récupération poids par défaut"""
        config = RewardShapingConfig()

        weights = config.get_weights()

        assert weights == RewardShapingConfig.DEFAULT

    def test_get_weights_custom(self):
        """Test récupération poids personnalisés"""
        config = RewardShapingConfig()
        custom_weights = {"punctuality_weight": 2.0, "distance_weight": 1.0}

        config.set_weights(custom_weights)
        weights = config.get_weights()

        assert weights == custom_weights

    def test_set_weights(self):
        """Test définition poids personnalisés"""
        config = RewardShapingConfig()
        custom_weights = {"punctuality_weight": 2.0, "distance_weight": 1.0}

        config.set_weights(custom_weights)

        assert config._custom_weights == custom_weights

    def test_set_weights_copy(self):
        """Test que set_weights fait une copie"""
        config = RewardShapingConfig()
        custom_weights = {"punctuality_weight": 2.0}

        config.set_weights(custom_weights)

        # Modifier l'original ne devrait pas affecter la copie
        custom_weights["punctuality_weight"] = 5.0

        assert config._custom_weights["punctuality_weight"] == 2.0

    def test_calculate_reward_comprehensive(self):
        """Test calcul de récompense complet avec tous les composants"""
        reward_shaping = AdvancedRewardShaping()

        state = np.random.rand(10)
        action = 1
        next_state = np.random.rand(10)
        info = {
            "is_late": False,
            "lateness_minutes": 0,
            "distance_km": 3.0,
            "driver_loads": [1, 1, 1],
            "assignment_successful": True,
            "assignment_time_minutes": 3,
            "driver_type": "REGULAR",
            "respects_preferences": True,
            "booking_priority": 3,
        }

        reward = reward_shaping.calculate_reward(state, action, next_state, info)

        # Vérifier que la récompense est positive et raisonnable
        assert isinstance(reward, float)
        assert reward > 0
        assert reward < 1000  # Limite raisonnable

    def test_calculate_reward_negative_scenario(self):
        """Test calcul de récompense avec scénario négatif"""
        reward_shaping = AdvancedRewardShaping()

        state = np.random.rand(10)
        action = 1
        next_state = np.random.rand(10)
        info = {
            "is_late": True,
            "lateness_minutes": 20,
            "is_outbound": True,
            "distance_km": 50.0,
            "driver_loads": [1, 10, 20],
            "assignment_successful": False,
            "driver_type": "EMERGENCY",
            "booking_priority": 2,
        }

        reward = reward_shaping.calculate_reward(state, action, next_state, info)

        # Vérifier que la récompense est négative
        assert isinstance(reward, float)
        assert reward < 0
