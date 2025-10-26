"""
Tests minimaux pour rl_logger.py - Version corrigée
"""
from unittest.mock import Mock, patch

import pytest

from services.rl.rl_logger import RLLogger


class TestRLLoggerMinimal:
    """Tests minimaux pour RLLogger"""

    def test_init_basic(self):
        """Test initialisation basique"""
        logger = RLLogger()

        assert logger.redis_key_prefix == "rl_logs"
        assert logger.max_redis_logs == 1000
        assert logger.enable_db_logging is True
        assert logger.enable_redis_logging is True

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés"""
        logger = RLLogger(
            redis_key_prefix="custom_prefix",
            max_redis_logs=0.500,
            enable_db_logging=False,
            enable_redis_logging=False
        )

        assert logger.redis_key_prefix == "custom_prefix"
        assert logger.max_redis_logs == 500
        assert logger.enable_db_logging is False
        assert logger.enable_redis_logging is False

    def test_hash_state(self):
        """Test hachage d'état"""
        logger = RLLogger()

        state = [1.0, 2.0, 3.0]
        state_hash = logger.hash_state(state)

        assert isinstance(state_hash, str)
        assert len(state_hash) == 40  # SHA-1 hash length

    def test_hash_state_with_different_inputs(self):
        """Test hachage d'état avec différents inputs"""
        logger = RLLogger()

        # Test avec liste
        state1 = [1.0, 2.0, 3.0]
        hash1 = logger.hash_state(state1)

        # Test avec tuple
        state2 = (1.0, 2.0, 3.0)
        hash2 = logger.hash_state(state2)

        # Les hash devraient être identiques pour le même contenu
        assert hash1 == hash2

    def test_log_decision_basic(self):
        """Test logging de décision basique"""
        logger = RLLogger()

        state = [1.0, 2.0, 3.0]
        action = 1
        q_values = [0.1, 0.8, 0.3]
        reward = 10.0
        latency_ms = 50
        model_version = "v1.0"
        constraints = {"max_wait": 30}
        metadata = {"test": True}

        # Mock les méthodes de logging pour éviter les erreurs DB/Redis
        with patch.object(logger, "_log_to_redis"), \
             patch.object(logger, "_log_to_db"):

            logger.log_decision(
                state, action, q_values, reward, latency_ms, model_version, constraints, metadata
            )

            # Vérifier que les méthodes ont été appelées
            logger._log_to_redis.assert_called_once()
            logger._log_to_db.assert_called_once()

    def test_log_decision_with_none_values(self):
        """Test logging de décision avec valeurs None"""
        logger = RLLogger()

        state = [1.0, 2.0, 3.0]
        action = 1
        q_values = [0.1, 0.8, 0.3]
        reward = 10.0
        latency_ms = 50
        model_version = "v1.0"
        constraints = None
        metadata = None

        # Mock les méthodes de logging
        with patch.object(logger, "_log_to_redis"), \
             patch.object(logger, "_log_to_db"):

            logger.log_decision(
                state, action, q_values, reward, latency_ms, model_version, constraints, metadata
            )

            # Devrait fonctionner sans erreur
            logger._log_to_redis.assert_called_once()
            logger._log_to_db.assert_called_once()

    def test_log_decision_with_empty_state(self):
        """Test logging de décision avec état vide"""
        logger = RLLogger()

        state = []
        action = 0
        q_values = []
        reward = 0.0
        latency_ms = 0
        model_version = "v1.0"
        constraints = {}
        metadata = {}

        # Mock les méthodes de logging
        with patch.object(logger, "_log_to_redis"), \
             patch.object(logger, "_log_to_db"):

            logger.log_decision(
                state, action, q_values, reward, latency_ms, model_version, constraints, metadata
            )

            # Devrait fonctionner sans erreur
            logger._log_to_redis.assert_called_once()
            logger._log_to_db.assert_called_once()

    def test_log_decision_with_large_data(self):
        """Test logging de décision avec données importantes"""
        logger = RLLogger()

        state = [1.0] * 1000  # État avec 1000 éléments
        action = 1
        q_values = [0.1] * 100  # 100 actions
        reward = 1000.0
        latency_ms = 1000
        model_version = "v1.0"
        constraints = {"key": "value" * 100}  # Contraintes importantes
        metadata = {"data": list(range(1000))}  # Métadonnées importantes

        # Mock les méthodes de logging
        with patch.object(logger, "_log_to_redis"), \
             patch.object(logger, "_log_to_db"):

            logger.log_decision(
                state, action, q_values, reward, latency_ms, model_version, constraints, metadata
            )

            # Devrait fonctionner sans erreur
            logger._log_to_redis.assert_called_once()
            logger._log_to_db.assert_called_once()

    def test_log_decision_with_nan_values(self):
        """Test logging de décision avec valeurs NaN"""
        logger = RLLogger()

        state = [1.0, float("nan"), 3.0]
        action = 1
        q_values = [0.1, float("nan"), 0.3]
        reward = float("nan")
        latency_ms = 50
        model_version = "v1.0"
        constraints = {}
        metadata = {}

        # Mock les méthodes de logging
        with patch.object(logger, "_log_to_redis"), \
             patch.object(logger, "_log_to_db"):

            logger.log_decision(
                state, action, q_values, reward, latency_ms, model_version, constraints, metadata
            )

            # Devrait fonctionner sans erreur
            logger._log_to_redis.assert_called_once()
            logger._log_to_db.assert_called_once()

    def test_log_decision_with_inf_values(self):
        """Test logging de décision avec valeurs inf"""
        logger = RLLogger()

        state = [1.0, float("inf"), 3.0]
        action = 1
        q_values = [0.1, float("inf"), 0.3]
        reward = float("inf")
        latency_ms = 50
        model_version = "v1.0"
        constraints = {}
        metadata = {}

        # Mock les méthodes de logging
        with patch.object(logger, "_log_to_redis"), \
             patch.object(logger, "_log_to_db"):

            logger.log_decision(
                state, action, q_values, reward, latency_ms, model_version, constraints, metadata
            )

            # Devrait fonctionner sans erreur
            logger._log_to_redis.assert_called_once()
            logger._log_to_db.assert_called_once()

    def test_log_decision_with_negative_values(self):
        """Test logging de décision avec valeurs négatives"""
        logger = RLLogger()

        state = [-1.0, -2.0, -3.0]
        action = -1
        q_values = [-0.1, -0.8, -0.3]
        reward = -10.0
        latency_ms = -50
        model_version = "v1.0"
        constraints = {}
        metadata = {}

        # Mock les méthodes de logging
        with patch.object(logger, "_log_to_redis"), \
             patch.object(logger, "_log_to_db"):

            logger.log_decision(
                state, action, q_values, reward, latency_ms, model_version, constraints, metadata
            )

            # Devrait fonctionner sans erreur
            logger._log_to_redis.assert_called_once()
            logger._log_to_db.assert_called_once()

    def test_get_stats(self):
        """Test récupération des statistiques"""
        logger = RLLogger()

        stats = logger.get_stats()

        assert isinstance(stats, dict)
        assert "total_logs" in stats
        assert "uptime_seconds" in stats
        assert "logs_per_second" in stats
        assert "success_rate" in stats

    def test_get_recent_logs(self):
        """Test récupération des logs récents"""
        logger = RLLogger()

        logs = logger.get_recent_logs(limit=10)

        assert isinstance(logs, list)

    def test_get_recent_logs_with_limit(self):
        """Test récupération des logs récents avec limite"""
        logger = RLLogger()

        logs = logger.get_recent_logs(limit=5)

        assert isinstance(logs, list)
        assert len(logs) <= 5

    def test_log_decision_with_exception(self):
        """Test logging de décision avec exception"""
        logger = RLLogger()

        state = [1.0, 2.0, 3.0]
        action = 1
        q_values = [0.1, 0.8, 0.3]
        reward = 10.0
        latency_ms = 50
        model_version = "v1.0"
        constraints = {}
        metadata = {}

        # Mock les méthodes pour lever une exception
        with patch.object(logger, "_log_to_redis", side_effect=Exception("Redis error")), \
             patch.object(logger, "_log_to_db", side_effect=Exception("DB error")):

            # Devrait gérer l'exception sans planter
            logger.log_decision(
                state, action, q_values, reward, latency_ms, model_version, constraints, metadata
            )

            # Les méthodes devraient avoir été appelées malgré l'exception
            logger._log_to_redis.assert_called_once()
            logger._log_to_db.assert_called_once()

    def test_log_decision_disabled_logging(self):
        """Test logging de décision avec logging désactivé"""
        logger = RLLogger(enable_db_logging=False, enable_redis_logging=False)

        state = [1.0, 2.0, 3.0]
        action = 1
        q_values = [0.1, 0.8, 0.3]
        reward = 10.0
        latency_ms = 50
        model_version = "v1.0"
        constraints = {}
        metadata = {}

        # Mock les méthodes pour vérifier qu'elles ne sont pas appelées
        with patch.object(logger, "_log_to_redis") as mock_redis, \
             patch.object(logger, "_log_to_db") as mock_db:

            logger.log_decision(
                state, action, q_values, reward, latency_ms, model_version, constraints, metadata
            )

            # Les méthodes ne devraient pas être appelées
            mock_redis.assert_not_called()
            mock_db.assert_not_called()
