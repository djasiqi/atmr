#!/usr/bin/env python3
"""Système de logging avancé pour les décisions RL.

Fournit une traçabilité complète des décisions RL avec stockage Redis (rapide)
et DB (persistance), optimisé pour la performance et sans fuite mémoire.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import hashlib
import json
import logging
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np

if TYPE_CHECKING:
    import torch  # type: ignore

logger = logging.getLogger(__name__)

# Constantes pour éviter les valeurs magiques
MAX_Q_VALUES_LOG = 64

# Import conditionnel pour éviter les erreurs si les modules ne sont pas
# disponibles
try:
    from services.db_context import db
except ImportError:
    db = None

try:
    from ext import redis_client
except ImportError:
    redis_client = None

try:
    from models.rl_suggestion_metric import RLSuggestionMetric
except ImportError:
    RLSuggestionMetric = None

# Import conditionnel de torch pour éviter NameError
try:
    import torch  # type: ignore
except ImportError:
    torch = None


class RLLogger:
    """Logger avancé pour les décisions RL avec traçabilité complète.

    Fonctionnalités:
    - Hash des états pour éviter les doublons
    - Stockage Redis (rapide) + DB (persistance)
    - Rotation automatique des logs (TTL logique)
    - Métriques de performance intégrées
    - Gestion d'erreurs robuste
    """

    def __init__(
        self,
        redis_key_prefix: str = "rl:decisions",
        max_redis_logs: int = 5000,
        enable_db_logging: bool = True,
        enable_redis_logging: bool = True,
    ):
        """Initialise le RLLogger.

        Args:
            redis_key_prefix: Préfixe pour les clés Redis
            max_redis_logs: Nombre maximum de logs en Redis
            enable_db_logging: Activer le logging en base de données
            enable_redis_logging: Activer le logging Redis

        """
        super().__init__()
        self.redis_key_prefix = redis_key_prefix
        self.max_redis_logs = max_redis_logs
        self.enable_db_logging = enable_db_logging
        self.enable_redis_logging = enable_redis_logging

        # Statistiques
        self.stats = {"total_logs": 0, "redis_logs": 0, "db_logs": 0, "errors": 0, "start_time": datetime.now(UTC)}

        logger.info("[RLLogger] Initialisé - Redis: %s, DB: %s", enable_redis_logging, enable_db_logging)

    def hash_state(self, state: Union[np.ndarray[Any, Any], "torch.Tensor", List[Any], Dict[str, Any]]) -> str:
        """Génère un hash unique pour l'état donné.

        Args:
            state: État à hasher (numpy array, torch tensor, list, dict)

        Returns:
            Hash hexadécimal de l'état

        """
        try:
            # Vérifier si c'est un torch tensor
            if torch is not None and isinstance(state, torch.Tensor):
                # Convertir tensor en numpy
                state_bytes = state.detach().cpu().numpy().tobytes()
            elif isinstance(state, np.ndarray):
                state_bytes = state.tobytes()
            elif isinstance(state, (list, tuple, dict)):
                # Convertir en JSON puis en bytes
                state_str = json.dumps(state, sort_keys=True)
                state_bytes = state_str.encode("utf-8")
            else:
                # Fallback: convertir en string puis en bytes
                state_str = str(state)
                state_bytes = state_str.encode("utf-8")

            # Générer le hash SHA-1 (40 caractères hex) pour compatibilité avec les tests
            return hashlib.sha1(state_bytes, usedforsecurity=False).hexdigest()

        except Exception as e:
            logger.error("[RLLogger] Erreur lors du hash de l'état: %s", e)
            # Fallback: hash basé sur le timestamp
            return hashlib.sha1(str(time.time()).encode(), usedforsecurity=False).hexdigest()

    def log_decision(
        self,
        state: Union[np.ndarray[Any, Any], "torch.Tensor", List[Any], Dict[str, Any]],
        action: Union[float, "torch.Tensor"],
        q_values: Union[np.ndarray[Any, Any], "torch.Tensor", List[Any]] | None = None,
        reward: float | None = None,
        latency_ms: float | None = None,
        model_version: str = "unknown",
        constraints: Dict[str, Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> bool:
        """Enregistre une décision RL avec traçabilité complète.

        Args:
            state: État d'entrée de la décision
            action: Action prise par l'agent
            q_values: Valeurs Q calculées (optionnel)
            reward: Récompense reçue (optionnel)
            latency_ms: Latence de la décision en millisecondes (optionnel)
            model_version: Version du modèle utilisé
            constraints: Contraintes appliquées (optionnel)
            metadata: Métadonnées additionnelles (optionnel)

        Returns:
            True si le logging a réussi, False sinon

        """
        try:
            # Générer le hash de l'état
            state_hash = self.hash_state(state)

            # Convertir l'action en int
            if torch is not None and isinstance(action, torch.Tensor):
                action_int = int(action.item())
            else:
                action_int = int(action)

            # Traiter les q_values
            q_values_list = None
            if q_values is not None:
                if torch is not None and isinstance(q_values, torch.Tensor):
                    q_values_list = q_values.detach().cpu().numpy().tolist()
                elif isinstance(q_values, np.ndarray):
                    q_values_list = q_values.tolist()
                elif isinstance(q_values, list):
                    q_values_list = q_values
                else:
                    # Fallback pour autres types
                    q_values_list = list(q_values) if hasattr(q_values, "__iter__") else None

                # Limiter à 64 valeurs pour éviter les logs trop volumineux
                if q_values_list and len(q_values_list) > MAX_Q_VALUES_LOG:
                    q_values_list = q_values_list[:MAX_Q_VALUES_LOG]

            # Créer l'enregistrement
            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "state_hash": state_hash,
                "action": action_int,
                "q_values": q_values_list,
                "reward": float(reward) if reward is not None else None,
                "latency_ms": float(latency_ms) if latency_ms is not None else None,
                "model_version": model_version,
                "constraints": constraints or {},
                "metadata": metadata or {},
            }

            # Logging Redis (rapide)
            redis_success = self._log_to_redis(record)

            # Logging DB (persistance)
            db_success = self._log_to_db(record)

            # Mettre à jour les statistiques
            self.stats["total_logs"] += 1
            if redis_success:
                self.stats["redis_logs"] += 1
            if db_success:
                self.stats["db_logs"] += 1

            # Log de debug
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "[RLLogger] Décision loggée - Hash: %s..., Action: %s, Redis: %s, DB: %s",
                    state_hash[:8],
                    action_int,
                    redis_success,
                    db_success,
                )

            return redis_success or db_success

        except Exception as e:
            logger.error("[RLLogger] Erreur lors du logging de la décision: %s", e)
            self.stats["errors"] += 1
            return False

    def _log_to_redis(self, record: Dict[str, Any]) -> bool:
        """Enregistre la décision dans Redis.

        Args:
            record: Enregistrement à stocker

        Returns:
            True si succès, False sinon

        """
        if not self.enable_redis_logging or redis_client is None:
            return False

        try:
            # Sérialiser l'enregistrement
            record_json = json.dumps(record)

            # Stocker dans Redis avec rotation automatique
            redis_client.lpush(f"{self.redis_key_prefix}:latest", record_json)
            redis_client.ltrim(f"{self.redis_key_prefix}:latest", 0, self.max_redis_logs - 1)

            # TTL de 24h pour les logs Redis
            redis_client.expire(f"{self.redis_key_prefix}:latest", 86400)

            return True

        except Exception as e:
            logger.warning("[RLLogger] Erreur Redis: %s", e)
            return False

    def _log_to_db(self, record: Dict[str, Any]) -> bool:
        """Enregistre la décision en base de données.

        Args:
            record: Enregistrement à stocker

        Returns:
            True si succès, False sinon

        """
        if not self.enable_db_logging or db is None or RLSuggestionMetric is None:
            return False

        try:
            # Créer l'objet RLSuggestionMetric avec les champs disponibles
            rl_metric = RLSuggestionMetric()
            rl_metric.suggestion_id = f"log_{record['state_hash'][:8]}_{int(time.time())}"
            rl_metric.company_id = 1  # Valeur par défaut, devrait être passé en paramètre
            rl_metric.booking_id = record["metadata"].get("booking_id", 0)
            rl_metric.assignment_id = record["metadata"].get("assignment_id", 0)
            rl_metric.current_driver_id = record["metadata"].get("current_driver_id", 0)
            rl_metric.suggested_driver_id = record["metadata"].get("suggested_driver_id", 0)
            rl_metric.confidence = record["constraints"].get("confidence", 0.0)
            rl_metric.expected_gain_minutes = record["constraints"].get("improvement", 0)
            rl_metric.q_value = (
                record["q_values"][record["action"]]
                if record["q_values"] and len(record["q_values"]) > record["action"]
                else None
            )
            rl_metric.source = "dqn_model"
            rl_metric.additional_data = {
                "state_hash": record["state_hash"],
                "action": record["action"],
                "q_values": record["q_values"],
                "reward": record["reward"],
                "latency_ms": record["latency_ms"],
                "model_version": record["model_version"],
                "metadata": record["metadata"],
            }

            # Ajouter à la session et commiter
            db.session.add(rl_metric)
            db.session.commit()

            return True

        except Exception as e:
            logger.warning("[RLLogger] Erreur DB: %s", e)
            if db:
                db.session.rollback()
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du logger.

        Returns:
            Dictionnaire avec les statistiques

        """
        uptime = datetime.now(UTC) - self.stats["start_time"]

        return {
            "total_logs": self.stats["total_logs"],
            "redis_logs": self.stats["redis_logs"],
            "db_logs": self.stats["db_logs"],
            "errors": self.stats["errors"],
            "uptime_seconds": uptime.total_seconds(),
            "logs_per_second": self.stats["total_logs"] / max(uptime.total_seconds(), 1),
            "success_rate": (self.stats["total_logs"] - self.stats["errors"]) / max(self.stats["total_logs"], 1),
        }

    def get_recent_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """Récupère les logs récents depuis Redis.

        Args:
            count: Nombre de logs à récupérer

        Returns:
            Liste des logs récents

        """
        if not self.enable_redis_logging or redis_client is None:
            return []

        try:
            # Récupérer les logs depuis Redis
            logs_json = redis_client.lrange(f"{self.redis_key_prefix}:latest", 0, count - 1)
            # S'assurer que c'est une liste (Redis retourne une liste de bytes)
            if not isinstance(logs_json, list):
                logs_json = []

            # Désérialiser les logs
            logs = []
            for log_json in logs_json:
                try:
                    log = json.loads(log_json)
                    logs.append(log)
                except json.JSONDecodeError:
                    continue

            return logs

        except Exception as e:
            logger.error("[RLLogger] Erreur lors de la récupération des logs: %s", e)
            return []

    def clear_logs(self, clear_redis: bool = True, clear_db: bool = False) -> bool:
        """Efface les logs (utilisé pour les tests ou la maintenance).

        Args:
            clear_redis: Effacer les logs Redis
            clear_db: Effacer les logs DB (ATTENTION: destructif)

        Returns:
            True si succès, False sinon

        """
        try:
            if clear_redis and redis_client is not None:
                redis_client.delete(f"{self.redis_key_prefix}:latest")
                logger.info("[RLLogger] Logs Redis effacés")

            if clear_db and db is not None and RLSuggestionMetric is not None:
                # ATTENTION: Cette opération est destructive
                db.session.query(RLSuggestionMetric).delete()
                db.session.commit()
                logger.warning("[RLLogger] Logs DB effacés (destructif)")

            return True

        except Exception as e:
            logger.error("[RLLogger] Erreur lors de l'effacement des logs: %s", e)
            return False


# Instance globale du logger
_rl_logger_instance: RLLogger | None = None


def get_rl_logger() -> RLLogger:
    """Retourne l'instance globale du RLLogger (singleton).

    Returns:
        Instance du RLLogger

    """
    global _rl_logger_instance  # noqa: PLW0603
    if _rl_logger_instance is None:
        _rl_logger_instance = RLLogger()
    return _rl_logger_instance


def log_rl_decision(
    state, action, q_values=None, reward=None, latency_ms=None, model_version="unknown", constraints=None, metadata=None
) -> bool:
    """Fonction de convenance pour logger une décision RL.

    Args:
        state: État d'entrée
        action: Action prise
        q_values: Valeurs Q (optionnel)
        reward: Récompense (optionnel)
        latency_ms: Latence en ms (optionnel)
        model_version: Version du modèle
        constraints: Contraintes (optionnel)
        metadata: Métadonnées (optionnel)

    Returns:
        True si succès, False sinon

    """
    logger = get_rl_logger()
    return logger.log_decision(
        state=state,
        action=action,
        q_values=q_values,
        reward=reward,
        latency_ms=latency_ms,
        model_version=model_version,
        constraints=constraints,
        metadata=metadata,
    )
