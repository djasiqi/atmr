#!/usr/bin/env python3
"""
Tests complets pour le système RLLogger.

Teste la traçabilité complète des décisions RL avec Redis et DB,
la performance du logging, et la gestion d'erreurs.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import json
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import conditionnel de torch
try:
    import torch  # type: ignore
except ImportError:
    torch = None

# Import conditionnel pour éviter les erreurs si les modules ne sont pas disponibles
try:
    from services.rl.rl_logger import RLLogger, get_rl_logger, log_rl_decision
except ImportError:
    RLLogger = None
    get_rl_logger = None
    log_rl_decision = None


class TestRLLogger:
    """Tests pour RLLogger."""

    @pytest.fixture
    def rl_logger(self):
        """Crée une instance de RLLogger pour les tests."""
        if RLLogger is None:
            pytest.skip("RLLogger non disponible")

        # Logger avec DB et Redis désactivés pour les tests
        return RLLogger(enable_db_logging=False, enable_redis_logging=False)

    @pytest.fixture
    def sample_state(self):
        """État d'exemple pour les tests."""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    @pytest.fixture
    def sample_q_values(self):
        """Q-values d'exemple pour les tests."""
        return np.array([0.1, 0.8, 0.3, 0.5, 0.2])

    def test_rl_logger_initialization(self):
        """Test l'initialisation du RLLogger."""
        if RLLogger is None:
            pytest.skip("RLLogger non disponible")

        logger = RLLogger()

        assert logger.redis_key_prefix == "rl:decisions"
        assert logger.max_redis_logs == 5000
        assert logger.enable_db_logging is True
        assert logger.enable_redis_logging is True

        # Vérifier les statistiques initiales
        stats = logger.get_stats()
        assert stats["total_logs"] == 0
        assert stats["redis_logs"] == 0
        assert stats["db_logs"] == 0
        assert stats["errors"] == 0

    def test_hash_state_numpy(self, rl_logger, sample_state):
        """Test le hash d'un état numpy."""
        hash1 = rl_logger.hash_state(sample_state)
        hash2 = rl_logger.hash_state(sample_state)

        # Même état = même hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 = 64 caractères hex

        # État différent = hash différent
        different_state = sample_state + 1
        hash3 = rl_logger.hash_state(different_state)
        assert hash1 != hash3

    def test_hash_state_torch_tensor(self, rl_logger):
        """Test le hash d'un tensor PyTorch."""
        if torch is None:
            pytest.skip("PyTorch non disponible")

        tensor = torch.tensor([1.0, 2.0, 3.0])
        hash1 = rl_logger.hash_state(tensor)

        assert len(hash1) == 64
        assert isinstance(hash1, str)

    def test_hash_state_list(self, rl_logger):
        """Test le hash d'une liste."""
        state_list = [1.0, 2.0, 3.0, 4.0]
        hash1 = rl_logger.hash_state(state_list)

        assert len(hash1) == 64
        assert isinstance(hash1, str)

    def test_hash_state_dict(self, rl_logger):
        """Test le hash d'un dictionnaire."""
        state_dict = {"feature1": 1.0, "feature2": 2.0}
        hash1 = rl_logger.hash_state(state_dict)

        assert len(hash1) == 64
        assert isinstance(hash1, str)

    def test_log_decision_basic(self, rl_logger, sample_state):
        """Test le logging basique d'une décision."""
        success = rl_logger.log_decision(
            state=sample_state,
            action=1,
            q_values=[0.1, 0.8, 0.3],
            reward=0.5,
            latency_ms=10.0,
            model_version="test_v1",
            constraints={"epsilon": 0.1},
            metadata={"test": True},
        )

        assert success is True

        # Vérifier les statistiques
        stats = rl_logger.get_stats()
        assert stats["total_logs"] == 1
        assert stats["errors"] == 0

    def test_log_decision_without_optional_params(self, rl_logger, sample_state):
        """Test le logging avec paramètres optionnels manquants."""
        success = rl_logger.log_decision(state=sample_state, action=2)

        assert success is True

        stats = rl_logger.get_stats()
        assert stats["total_logs"] == 1

    def test_log_decision_error_handling(self, rl_logger):
        """Test la gestion d'erreurs lors du logging."""
        # Test avec des données invalides
        success = rl_logger.log_decision(
            state="invalid_state",  # Type invalide
            action="invalid_action",  # Type invalide
        )

        # Doit gérer l'erreur gracieusement
        assert success is False

        stats = rl_logger.get_stats()
        assert stats["errors"] == 1

    def test_get_stats(self, rl_logger, sample_state):
        """Test la récupération des statistiques."""
        # Loguer quelques décisions
        for i in range(3):
            rl_logger.log_decision(state=sample_state, action=i, model_version=f"test_v{i}")

        stats = rl_logger.get_stats()

        assert stats["total_logs"] == 3
        assert stats["uptime_seconds"] > 0
        assert stats["logs_per_second"] > 0
        assert stats["success_rate"] == 1.0

    def test_clear_logs(self, rl_logger, sample_state):
        """Test l'effacement des logs."""
        # Loguer quelques décisions
        rl_logger.log_decision(state=sample_state, action=1)
        rl_logger.log_decision(state=sample_state, action=2)

        # Effacer les logs
        success = rl_logger.clear_logs(clear_redis=True, clear_db=False)
        assert success is True

    def test_performance_under_load(self, rl_logger, sample_state):
        """Test les performances sous charge."""
        start_time = time.time()

        # Loguer 100 décisions
        for i in range(100):
            rl_logger.log_decision(state=sample_state, action=i, latency_ms=i * 0.1)

        end_time = time.time()
        total_time = end_time - start_time

        # Vérifier que chaque log prend moins de 1ms en moyenne
        avg_time_per_log = total_time / 100
        assert avg_time_per_log < 0.0001  # 1ms

        stats = rl_logger.get_stats()
        assert stats["total_logs"] == 100


class TestRLLoggerWithRedis:
    """Tests pour RLLogger avec Redis."""

    @pytest.fixture
    def rl_logger_redis(self):
        """Crée une instance de RLLogger avec Redis activé."""
        if RLLogger is None:
            pytest.skip("RLLogger non disponible")

        return RLLogger(enable_db_logging=False, enable_redis_logging=True)

    def test_redis_logging(self, rl_logger_redis, sample_state):
        """Test le logging Redis."""
        # Mock Redis pour éviter les erreurs de connexion
        with patch("services.rl.rl_logger.redis_client") as mock_redis:
            mock_redis.lpush.return_value = 1
            mock_redis.ltrim.return_value = True
            mock_redis.expire.return_value = True

            success = rl_logger_redis.log_decision(state=sample_state, action=1, model_version="test_redis")

            assert success is True
            mock_redis.lpush.assert_called_once()
            mock_redis.ltrim.assert_called_once()
            mock_redis.expire.assert_called_once()

    def test_get_recent_logs(self, rl_logger_redis, sample_state):
        """Test la récupération des logs récents."""
        with patch("services.rl.rl_logger.redis_client") as mock_redis:
            # Mock des logs Redis
            mock_logs = [
                json.dumps({"action": 1, "state_hash": "abc123"}),
                json.dumps({"action": 2, "state_hash": "def456"}),
            ]
            mock_redis.lrange.return_value = mock_logs

            logs = rl_logger_redis.get_recent_logs(count=10)

            assert len(logs) == 2
            assert logs[0]["action"] == 1
            assert logs[1]["action"] == 2


class TestRLLoggerWithDB:
    """Tests pour RLLogger avec base de données."""

    @pytest.fixture
    def rl_logger_db(self):
        """Crée une instance de RLLogger avec DB activée."""
        if RLLogger is None:
            pytest.skip("RLLogger non disponible")

        return RLLogger(enable_db_logging=True, enable_redis_logging=False)

    def test_db_logging(self, rl_logger_db, sample_state):
        """Test le logging en base de données."""
        # Mock des modules DB
        with (
            patch("services.rl.rl_logger.db") as mock_db,
            patch("services.rl.rl_logger.RLSuggestionMetric") as mock_metric,
        ):
            mock_session = Mock()
            mock_db.session = mock_session

            mock_metric_instance = Mock()
            mock_metric.return_value = mock_metric_instance

            success = rl_logger_db.log_decision(state=sample_state, action=1, model_version="test_db")

            assert success is True
            mock_db.session.add.assert_called_once()
            mock_db.session.commit.assert_called_once()


class TestRLLoggerIntegration:
    """Tests d'intégration pour RLLogger."""

    def test_get_rl_logger_singleton(self):
        """Test le singleton RLLogger."""
        if get_rl_logger is None:
            pytest.skip("get_rl_logger non disponible")

        logger1 = get_rl_logger()
        logger2 = get_rl_logger()

        assert logger1 is logger2
        assert isinstance(logger1, RLLogger) if RLLogger is not None else True

    def test_log_rl_decision_convenience_function(self, sample_state):
        """Test la fonction de convenance log_rl_decision."""
        if log_rl_decision is None:
            pytest.skip("log_rl_decision non disponible")

        with patch("services.rl.rl_logger.get_rl_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_logger.log_decision.return_value = True
            mock_get_logger.return_value = mock_logger

            success = log_rl_decision(state=sample_state, action=1, model_version="test_convenience")

            assert success is True
            mock_logger.log_decision.assert_called_once()

    def test_logging_with_torch_tensors(self, sample_state):
        """Test le logging avec des tensors PyTorch."""
        if torch is None or RLLogger is None:
            pytest.skip("PyTorch ou RLLogger non disponible")

        logger = RLLogger(enable_db_logging=False, enable_redis_logging=False)

        # Créer des tensors PyTorch
        state_tensor = torch.tensor(sample_state)
        q_values_tensor = torch.tensor([0.1, 0.8, 0.3, 0.5, 0.2])

        success = logger.log_decision(
            state=state_tensor, action=1, q_values=q_values_tensor, model_version="test_torch"
        )

        assert success is True

    def test_error_recovery(self, sample_state):
        """Test la récupération après erreurs."""
        if RLLogger is None:
            pytest.skip("RLLogger non disponible")

        logger = RLLogger(enable_db_logging=False, enable_redis_logging=False)

        # Première décision avec erreur
        success1 = logger.log_decision(
            state="invalid",  # Erreur
            action=1,
        )
        assert success1 is False

        # Deuxième décision normale
        success2 = logger.log_decision(state=sample_state, action=2)
        assert success2 is True

        stats = logger.get_stats()
        assert stats["total_logs"] == 2
        assert stats["errors"] == 1
        assert stats["success_rate"] == 0.5


def run_rl_logger_tests():
    """Exécute tous les tests RLLogger."""
    print("🧪 Exécution des tests RLLogger")

    # Tests de base
    test_classes = [TestRLLogger, TestRLLoggerWithRedis, TestRLLoggerWithDB, TestRLLoggerIntegration]

    total_tests = 0
    passed_tests = 0

    for test_class in test_classes:
        print("\n📋 Tests {test_class.__name__}")

        # Créer une instance de la classe de test
        test_instance = test_class()

        # Exécuter les méthodes de test
        for method_name in dir(test_instance):
            if method_name.startswith("test_"):
                total_tests += 1
                try:
                    method = getattr(test_instance, method_name)
                    method()
                    print("  ✅ {method_name}")
                    passed_tests += 1
                except Exception:
                    print("  ❌ {method_name}: {e}")

    print("\n📊 Résultats des tests RLLogger:")
    print("  Tests exécutés: {total_tests}")
    print("  Tests réussis: {passed_tests}")
    print("  Taux de succès: {passed_tests/total_tests*100" if total_tests > 0 else "  Taux de succès: 0%")

    return passed_tests, total_tests


if __name__ == "__main__":
    run_rl_logger_tests()
