#!/usr/bin/env python3
"""
Tests pour rl_logger.py - couverture de base
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from services.rl.rl_logger import RLLogger


class TestRLLogger:
    """Tests pour la classe RLLogger."""

    def test_init_with_default_params(self):
        """Test initialisation avec paramètres par défaut."""
        logger = RLLogger()

        assert logger.redis_key_prefix == "rl:decisions"
        assert logger.max_redis_logs == 5000
        assert logger.enable_db_logging is True
        assert logger.enable_redis_logging is True
        assert logger.stats["total_logs"] == 0
        assert logger.stats["redis_logs"] == 0
        assert logger.stats["db_logs"] == 0
        assert logger.stats["errors"] == 0

    def test_init_with_custom_params(self):
        """Test initialisation avec paramètres personnalisés."""
        logger = RLLogger(
            redis_key_prefix="custom:rl",
            max_redis_logs=0.1000,
            enable_db_logging=False,
            enable_redis_logging=False
        )

        assert logger.redis_key_prefix == "custom:rl"
        assert logger.max_redis_logs == 1000
        assert logger.enable_db_logging is False
        assert logger.enable_redis_logging is False

    def test_hash_state_numpy_array(self):
        """Test hash_state avec numpy array."""
        logger = RLLogger()

        state = np.array([1, 2, 3, 4, 5])
        hash_result = logger.hash_state(state)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 40  # SHA-1 hash length

    def test_hash_state_list(self):
        """Test hash_state avec liste."""
        logger = RLLogger()

        state = [1, 2, 3, 4, 5]
        hash_result = logger.hash_state(state)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 40

    def test_hash_state_dict(self):
        """Test hash_state avec dictionnaire."""
        logger = RLLogger()

        state = {"key1": 1, "key2": 2, "key3": 3}
        hash_result = logger.hash_state(state)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 40

    def test_hash_state_empty(self):
        """Test hash_state avec état vide."""
        logger = RLLogger()

        state = np.array([])
        hash_result = logger.hash_state(state)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 40

    def test_hash_state_with_nan(self):
        """Test hash_state avec valeurs NaN."""
        logger = RLLogger()

        state = np.array([1, 2, np.nan, 4, 5])
        hash_result = logger.hash_state(state)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 40

    def test_hash_state_with_inf(self):
        """Test hash_state avec valeurs infinies."""
        logger = RLLogger()

        state = np.array([1, 2, np.inf, 4, 5])
        hash_result = logger.hash_state(state)

        assert isinstance(hash_result, str)
        assert len(hash_result) == 40

    def test_log_decision_basic(self):
        """Test log_decision avec paramètres de base."""
        logger = RLLogger()

        # Données de test
        state = np.array([1, 2, 3, 4, 5])
        action = 2
        q_values = np.array([0.1, 0.2, 0.9, 0.3])
        reward = 1.5

        # Test sans erreur
        result = logger.log_decision(
            state=state,
            action=action,
            q_values=q_values,
            reward=reward
        )

        assert result is True

    def test_log_decision_with_metadata(self):
        """Test log_decision avec métadonnées."""
        logger = RLLogger()

        # Données de test
        state = np.array([1, 2, 3, 4, 5])
        action = 2
        q_values = np.array([0.1, 0.2, 0.9, 0.3])
        reward = 1.5
        metadata = {"test": "data", "version": "1.0"}

        result = logger.log_decision(
            state=state,
            action=action,
            q_values=q_values,
            reward=reward,
            metadata=metadata
        )

        assert result is True

    def test_log_decision_with_constraints(self):
        """Test log_decision avec contraintes."""
        logger = RLLogger()

        # Données de test
        state = np.array([1, 2, 3, 4, 5])
        action = 2
        q_values = np.array([0.1, 0.2, 0.9, 0.3])
        reward = 1.5
        constraints = {"max_delay": 30, "capacity": 4}

        result = logger.log_decision(
            state=state,
            action=action,
            q_values=q_values,
            reward=reward,
            constraints=constraints
        )

        assert result is True

    def test_log_decision_with_latency(self):
        """Test log_decision avec latence."""
        logger = RLLogger()

        # Données de test
        state = np.array([1, 2, 3, 4, 5])
        action = 2
        q_values = np.array([0.1, 0.2, 0.9, 0.3])
        reward = 1.5
        latency_ms = 25.5

        result = logger.log_decision(
            state=state,
            action=action,
            q_values=q_values,
            reward=reward,
            latency_ms=latency_ms
        )

        assert result is True

    def test_log_decision_empty_state(self):
        """Test log_decision avec état vide."""
        logger = RLLogger()

        # Données de test avec état vide
        state = np.array([])
        action = 0
        q_values = np.array([0.1])
        reward = 0.0

        result = logger.log_decision(
            state=state,
            action=action,
            q_values=q_values,
            reward=reward
        )

        assert result is True

    def test_log_decision_large_data(self):
        """Test log_decision avec données importantes."""
        logger = RLLogger()

        # Données de test importantes
        state = np.random.rand(100)
        action = 50
        q_values = np.random.rand(100)
        reward = 100.0

        result = logger.log_decision(
            state=state,
            action=action,
            q_values=q_values,
            reward=reward
        )

        assert result is True

    def test_log_decision_nan_values(self):
        """Test log_decision avec valeurs NaN."""
        logger = RLLogger()

        # Données de test avec NaN
        state = np.array([1, 2, np.nan, 4, 5])
        action = 2
        q_values = np.array([0.1, np.nan, 0.9, 0.3])
        reward = np.nan

        result = logger.log_decision(
            state=state,
            action=action,
            q_values=q_values,
            reward=reward
        )

        assert result is True

    def test_log_decision_inf_values(self):
        """Test log_decision avec valeurs infinies."""
        logger = RLLogger()

        # Données de test avec inf
        state = np.array([1, 2, np.inf, 4, 5])
        action = 2
        q_values = np.array([0.1, np.inf, 0.9, 0.3])
        reward = np.inf

        result = logger.log_decision(
            state=state,
            action=action,
            q_values=q_values,
            reward=reward
        )

        assert result is True

    def test_log_decision_negative_values(self):
        """Test log_decision avec valeurs négatives."""
        logger = RLLogger()

        # Données de test avec valeurs négatives
        state = np.array([-1, -2, -3, -4, -5])
        action = 2
        q_values = np.array([-0.1, -0.2, -0.9, -0.3])
        reward = -10.0

        result = logger.log_decision(
            state=state,
            action=action,
            q_values=q_values,
            reward=reward
        )

        assert result is True

    def test_get_stats(self):
        """Test get_stats."""
        logger = RLLogger()

        stats = logger.get_stats()

        assert isinstance(stats, dict)
        assert "total_logs" in stats
        assert "redis_logs" in stats
        assert "db_logs" in stats
        assert "errors" in stats
        assert "uptime_seconds" in stats
        assert "logs_per_second" in stats
        assert "success_rate" in stats

    def test_get_recent_logs(self):
        """Test récupération des logs récents"""
        logger = RLLogger()

        logs = logger.get_recent_logs(count=10)

        assert isinstance(logs, list)
        assert len(logs) <= 10
