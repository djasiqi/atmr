#!/usr/bin/env python3

"""Système de Safety Guards pour le dispatch RL.

Ce module implémente des garde-fous critiques pour éviter les décisions
dangereuses du système RL et déclencher des rollbacks automatiques.

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np

# Constantes pour éviter les valeurs magiques
VIOLATION_COUNT_THRESHOLD = 5
TOTAL_VIOLATIONS_THRESHOLD = 10

logger = logging.getLogger(__name__)


@dataclass
class SafetyThresholds:
    """Seuils de sécurité configurables pour les Safety Guards.

    Ces seuils définissent les limites critiques au-delà desquelles
    une décision RL est considérée comme dangereuse.
    """

    # Seuils de performance
    max_delay_minutes: float = 30.0  # Retard maximum acceptable (minutes)
    invalid_action_rate: float = 0.03  # Taux maximum d'actions invalides (3%)
    min_completion_rate: float = 0.90  # Taux minimum de completion (90%)

    # Seuils de charge
    max_driver_load: int = 12  # Charge maximum par chauffeur
    min_driver_utilization: float = 0.60  # Utilisation minimum (60%)

    # Seuils de distance
    max_avg_distance_km: float = 25.0  # Distance moyenne maximum (km)
    max_single_distance_km: float = 50.0  # Distance single maximum (km)

    # Seuils de confiance RL
    min_rl_confidence: float = 0.70  # Confiance minimum RL (70%)
    max_uncertainty_threshold: float = 0.25  # Incertitude maximum (25%)

    # Seuils temporels
    max_decision_time_ms: float = 100.0  # Temps de décision maximum (ms)
    min_episode_length: int = 50  # Longueur minimum d'épisode

    # Seuils de variance
    max_reward_variance: float = 0.15  # Variance maximum des rewards (15%)
    max_q_value_drift: float = 0.20  # Dérive maximum des Q-values (20%)


class SafetyGuards:
    """Système de garde-fous pour le dispatch RL.

    Features:
    - Détection de décisions dangereuses
    - Rollback automatique vers heuristiques
    - Alertes proactives
    - Monitoring des métriques critiques
    - Logging détaillé des violations
    """

    def __init__(self, thresholds: SafetyThresholds | None = None):
        """Initialise les Safety Guards.

        Args:
            thresholds: Seuils de sécurité (utilise les valeurs par défaut si None)

        """
        super().__init__()
        self.thresholds = thresholds or SafetyThresholds()

        # Historique des violations pour pattern detection
        self.violation_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        # Compteurs de rollbacks
        self.rollback_count = 0
        self.last_rollback_time: datetime | None = None

        # Cache des métriques pour performance
        self.metrics_cache: Dict[str, Any] = {}
        self.cache_ttl_minutes = 5

        logger.info("[SafetyGuards] Initialisé avec seuils: %s", self.thresholds)

    def check_dispatch_result(
        self, dispatch_result: Dict[str, Any], rl_metadata: Dict[str, Any] | None = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Vérifie la sécurité d'un résultat de dispatch.

        Args:
            dispatch_result: Résultat du dispatch (assignations, métriques)
            rl_metadata: Métadonnées RL (confiance, Q-values, etc.)

        Returns:
            Tuple (is_safe, detailed_checks)

        """
        try:
            # Extraire les métriques du résultat
            metrics = self._extract_metrics(dispatch_result, rl_metadata)

            # Effectuer tous les checks de sécurité
            checks = self._perform_safety_checks(metrics)

            # Déterminer si le résultat est sûr
            is_safe = all(checks.values())

            # Logging détaillé
            if not is_safe:
                self._log_violations(checks, metrics)
                self._record_violation(checks, metrics)

            # Mettre à jour les métriques de performance
            self._update_performance_metrics(is_safe, checks)

            return is_safe, {
                "is_safe": is_safe,
                "checks": checks,
                "metrics": metrics,
                "violation_count": len([c for c in checks.values() if not c]),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        except Exception as e:
            logger.error("[SafetyGuards] Erreur lors du check: %s", e)
            # En cas d'erreur, considérer comme dangereux
            return False, {"is_safe": False, "error": str(e), "timestamp": datetime.now(UTC).isoformat()}

    def _extract_metrics(self, dispatch_result: Dict[str, Any], rl_metadata: Dict[str, Any] | None) -> Dict[str, Any]:
        """Extrait et normalise les métriques du résultat de dispatch."""
        metrics = {}

        # Métriques de performance
        metrics["max_delay_minutes"] = dispatch_result.get("max_delay_minutes", 0)
        metrics["avg_delay_minutes"] = dispatch_result.get("avg_delay_minutes", 0)
        metrics["completion_rate"] = dispatch_result.get("completion_rate", 1.0)
        metrics["invalid_action_rate"] = dispatch_result.get("invalid_action_rate", 0.0)

        # Métriques de charge
        driver_loads = dispatch_result.get("driver_loads", [])
        if driver_loads:
            metrics["max_driver_load"] = max(driver_loads)
            metrics["avg_driver_load"] = np.mean(driver_loads)
            metrics["driver_load_variance"] = np.var(driver_loads)
        else:
            metrics["max_driver_load"] = 0
            metrics["avg_driver_load"] = 0
            metrics["driver_load_variance"] = 0

        # Métriques de distance
        metrics["avg_distance_km"] = dispatch_result.get("avg_distance_km", 0)
        metrics["max_distance_km"] = dispatch_result.get("max_distance_km", 0)
        metrics["total_distance_km"] = dispatch_result.get("total_distance_km", 0)

        # Métriques RL si disponibles
        if rl_metadata:
            metrics["rl_confidence"] = rl_metadata.get("confidence", 0.0)
            metrics["rl_uncertainty"] = rl_metadata.get("uncertainty", 0.0)
            metrics["decision_time_ms"] = rl_metadata.get("decision_time_ms", 0)
            metrics["q_value_variance"] = rl_metadata.get("q_value_variance", 0)
            metrics["episode_length"] = rl_metadata.get("episode_length", 0)
        else:
            metrics["rl_confidence"] = 1.0  # Par défaut, confiance maximale
            metrics["rl_uncertainty"] = 0.0
            metrics["decision_time_ms"] = 0
            metrics["q_value_variance"] = 0
            metrics["episode_length"] = 0

        return metrics

    def _perform_safety_checks(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Effectue tous les checks de sécurité."""
        checks = {}

        # Check 1: Retards
        checks["max_delay_ok"] = metrics["max_delay_minutes"] <= self.thresholds.max_delay_minutes

        # Check 2: Actions invalides
        checks["invalid_actions_ok"] = metrics["invalid_action_rate"] <= self.thresholds.invalid_action_rate

        # Check 3: Taux de completion
        checks["completion_rate_ok"] = metrics["completion_rate"] >= self.thresholds.min_completion_rate

        # Check 4: Charge des chauffeurs
        checks["driver_load_ok"] = metrics["max_driver_load"] <= self.thresholds.max_driver_load

        # Check 5: Utilisation des chauffeurs
        driver_utilization = (
            metrics["avg_driver_load"] / max(1, metrics["max_driver_load"]) if metrics["max_driver_load"] > 0 else 1.0
        )
        checks["driver_utilization_ok"] = driver_utilization >= self.thresholds.min_driver_utilization

        # Check 6: Distances
        checks["avg_distance_ok"] = metrics["avg_distance_km"] <= self.thresholds.max_avg_distance_km
        checks["max_distance_ok"] = metrics["max_distance_km"] <= self.thresholds.max_single_distance_km

        # Check 7: Confiance RL
        checks["rl_confidence_ok"] = metrics["rl_confidence"] >= self.thresholds.min_rl_confidence

        # Check 8: Incertitude RL
        checks["rl_uncertainty_ok"] = metrics["rl_uncertainty"] <= self.thresholds.max_uncertainty_threshold

        # Check 9: Temps de décision
        checks["decision_time_ok"] = metrics["decision_time_ms"] <= self.thresholds.max_decision_time_ms

        # Check 10: Longueur d'épisode
        checks["episode_length_ok"] = metrics["episode_length"] >= self.thresholds.min_episode_length

        return checks

    def _log_violations(self, checks: Dict[str, bool], metrics: Dict[str, Any]) -> None:
        """Log les violations détectées."""
        violations = [check_name for check_name, passed in checks.items() if not passed]

        logger.warning("[SafetyGuards] ⚠️ Violations détectées: %s", ", ".join(violations))

        # Log détaillé des métriques problématiques
        for violation in violations:
            if violation == "max_delay_ok":
                logger.warning(
                    "[SafetyGuards] Retard critique: %.1f min > %.1f min",
                    metrics["max_delay_minutes"],
                    self.thresholds.max_delay_minutes,
                )
            elif violation == "invalid_actions_ok":
                logger.warning(
                    "[SafetyGuards] Actions invalides: %.3f > %.3f",
                    metrics["invalid_action_rate"],
                    self.thresholds.invalid_action_rate,
                )
            elif violation == "completion_rate_ok":
                logger.warning(
                    "[SafetyGuards] Completion rate: %.3f < %.3f",
                    metrics["completion_rate"],
                    self.thresholds.min_completion_rate,
                )
            elif violation == "rl_confidence_ok":
                logger.warning(
                    "[SafetyGuards] Confiance RL faible: %.3f < %.3f",
                    metrics["rl_confidence"],
                    self.thresholds.min_rl_confidence,
                )

    def _record_violation(self, checks: Dict[str, bool], metrics: Dict[str, Any]) -> None:
        """Enregistre la violation dans l'historique."""
        violation_record = {
            "timestamp": datetime.now(UTC),
            "violations": [check_name for check_name, passed in checks.items() if not passed],
            "metrics": metrics.copy(),
            "severity": self._calculate_severity(checks, metrics),
        }

        self.violation_history.append(violation_record)

        # Maintenir la taille de l'historique
        if len(self.violation_history) > self.max_history_size:
            self.violation_history = self.violation_history[-self.max_history_size :]

    def _calculate_severity(self, checks: Dict[str, bool], metrics: Dict[str, Any]) -> str:  # noqa: ARG002
        """Calcule la sévérité de la violation."""
        violation_count = sum(1 for passed in checks.values() if not passed)

        if violation_count >= VIOLATION_COUNT_THRESHOLD:
            return "CRITICAL"
        if violation_count >= VIOLATION_COUNT_THRESHOLD:
            return "HIGH"
        if violation_count >= VIOLATION_COUNT_THRESHOLD:
            return "MEDIUM"
        return "LOW"

    def _update_performance_metrics(self, is_safe: bool, checks: Dict[str, bool]) -> None:
        """Met à jour les métriques de performance des guards."""
        # Cette méthode peut être étendue pour suivre les performances
        # des guards eux-mêmes (faux positifs, etc.)

    def should_rollback(self, recent_violations: int = 3) -> bool:
        """Détermine si un rollback doit être déclenché.

        Args:
            recent_violations: Nombre de violations récentes pour déclencher rollback

        Returns:
            True si un rollback est recommandé

        """
        # Vérifier les violations récentes (dernières 30 minutes)
        cutoff_time = datetime.now(UTC) - timedelta(minutes=30)
        recent_violations_list = [v for v in self.violation_history if v["timestamp"] > cutoff_time]

        # Rollback si trop de violations récentes
        if len(recent_violations_list) >= recent_violations:
            logger.warning("[SafetyGuards] 🚨 Rollback recommandé: %d violations récentes", len(recent_violations_list))
            return True

        # Rollback si dernière violation était critique
        if recent_violations_list:
            last_violation = recent_violations_list[-1]
            if last_violation["severity"] == "CRITICAL":
                logger.warning("[SafetyGuards] 🚨 Rollback recommandé: violation CRITIQUE")
                return True

        return False

    def get_health_status(self) -> Dict[str, Any]:
        """Retourne le statut de santé des Safety Guards."""
        total_violations = len(self.violation_history)
        recent_violations = len(
            [v for v in self.violation_history if v["timestamp"] > datetime.now(UTC) - timedelta(hours=24)]
        )

        return {
            "status": "healthy" if total_violations < TOTAL_VIOLATIONS_THRESHOLD else "degraded",
            "total_violations": total_violations,
            "recent_violations_24h": recent_violations,
            "rollback_count": self.rollback_count,
            "last_rollback": self.last_rollback_time.isoformat() if self.last_rollback_time else None,
            "thresholds": self.thresholds.__dict__,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def update_thresholds(self, new_thresholds: Dict[str, Any]) -> None:
        """Met à jour les seuils de sécurité.

        Args:
            new_thresholds: Nouveaux seuils à appliquer

        """
        for key, value in new_thresholds.items():
            if hasattr(self.thresholds, key):
                setattr(self.thresholds, key, value)
                logger.info("[SafetyGuards] Seuil %s mis à jour: %s", key, value)
            else:
                logger.warning("[SafetyGuards] Seuil inconnu ignoré: %s", key)


# Instance globale des Safety Guards
_safety_guards_instance: SafetyGuards | None = None


def get_safety_guards() -> SafetyGuards:
    """Retourne l'instance globale des Safety Guards."""
    global _safety_guards_instance  # noqa: PLW0603
    if _safety_guards_instance is None:
        _safety_guards_instance = SafetyGuards()
    return _safety_guards_instance


def reset_safety_guards() -> None:
    """Remet à zéro l'instance globale des Safety Guards."""
    global _safety_guards_instance  # noqa: PLW0603
    _safety_guards_instance = None
