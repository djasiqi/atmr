# ruff: noqa: W293
"""
Système de feature flags pour activer/désactiver fonctionnalités en production.

Usage:
    from feature_flags import FeatureFlags
    
    if FeatureFlags.is_ml_enabled():
        prediction = ml_predictor.predict_delay(booking, driver)
    else:
        prediction = heuristic_fallback(booking)
"""
import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)

# Configuration via variables d'environnement
ML_ENABLED = os.getenv("ML_ENABLED", "false").lower() == "true"
ML_TRAFFIC_PERCENTAGE = int(os.getenv("ML_TRAFFIC_PERCENTAGE", "10"))  # 10% par défaut
FALLBACK_ON_ERROR = os.getenv("FALLBACK_ON_ERROR", "true").lower() == "true"


class FeatureFlags:
    """
    Gestion centralisée des feature flags.
    
    Permet d'activer/désactiver des fonctionnalités sans redéploiement.
    """

    # Configuration ML
    _ml_enabled = ML_ENABLED
    _ml_traffic_percentage = ML_TRAFFIC_PERCENTAGE
    _fallback_on_error = FALLBACK_ON_ERROR

    # Stats (pour monitoring)
    _ml_requests = 0
    _ml_successes = 0
    _ml_failures = 0
    _fallback_requests = 0

    @classmethod
    def is_ml_enabled(cls, request_id: str | None = None) -> bool:
        """
        Vérifie si le ML est activé pour cette requête.
        
        Args:
            request_id: ID de la requête (pour logging)
        
        Returns:
            True si ML activé pour cette requête, False sinon
        """
        if not cls._ml_enabled:
            logger.debug(f"[FeatureFlag] ML disabled globally (request: {request_id})")
            return False

        # Activation progressive basée sur pourcentage
        if cls._ml_traffic_percentage < 100:
            # Utiliser random pour distribuer le trafic
            use_ml = random.randint(1, 100) <= cls._ml_traffic_percentage

            if use_ml:
                cls._ml_requests += 1
                logger.info(
                    f"[FeatureFlag] ML enabled for request {request_id} "
                    f"({cls._ml_traffic_percentage}% traffic)"
                )
            else:
                cls._fallback_requests += 1
                logger.debug(
                    f"[FeatureFlag] ML skipped for request {request_id} "
                    f"(outside {cls._ml_traffic_percentage}% traffic)"
                )

            return use_ml

        # 100% du trafic
        cls._ml_requests += 1
        return True

    @classmethod
    def should_fallback_on_error(cls) -> bool:
        """
        Vérifie si on doit utiliser le fallback en cas d'erreur ML.
        
        Returns:
            True si fallback activé, False sinon
        """
        return cls._fallback_on_error

    @classmethod
    def record_ml_success(cls) -> None:
        """Enregistre une prédiction ML réussie."""
        cls._ml_successes += 1

    @classmethod
    def record_ml_failure(cls) -> None:
        """Enregistre une erreur de prédiction ML."""
        cls._ml_failures += 1

        # Auto-désactivation si taux d'erreur > 20%
        if cls._ml_requests > 100:  # Au moins 100 requêtes
            error_rate = cls._ml_failures / cls._ml_requests
            if error_rate > 0.20:
                logger.error(
                    f"[FeatureFlag] High ML error rate ({error_rate:.1%}), "
                    "consider disabling ML"
                )

    @classmethod
    def get_stats(cls) -> dict[str, Any]:
        """
        Retourne les statistiques d'utilisation.
        
        Returns:
            Dict avec stats ML et fallback
        """
        total_requests = cls._ml_requests + cls._fallback_requests

        return {
            "ml_enabled": cls._ml_enabled,
            "ml_traffic_percentage": cls._ml_traffic_percentage,
            "fallback_on_error": cls._fallback_on_error,
            "total_requests": total_requests,
            "ml_requests": cls._ml_requests,
            "ml_successes": cls._ml_successes,
            "ml_failures": cls._ml_failures,
            "ml_success_rate": (
                cls._ml_successes / cls._ml_requests
                if cls._ml_requests > 0
                else 0.0
            ),
            "fallback_requests": cls._fallback_requests,
            "ml_usage_rate": (
                cls._ml_requests / total_requests
                if total_requests > 0
                else 0.0
            ),
        }

    @classmethod
    def reset_stats(cls) -> None:
        """Réinitialise les statistiques."""
        cls._ml_requests = 0
        cls._ml_successes = 0
        cls._ml_failures = 0
        cls._fallback_requests = 0
        logger.info("[FeatureFlag] Stats reset")

    @classmethod
    def set_ml_enabled(cls, enabled: bool) -> None:
        """
        Active/désactive le ML globalement.
        
        Args:
            enabled: True pour activer, False pour désactiver
        """
        cls._ml_enabled = enabled
        logger.warning(f"[FeatureFlag] ML globally {'enabled' if enabled else 'disabled'}")

    @classmethod
    def set_ml_traffic_percentage(cls, percentage: int) -> None:
        """
        Configure le pourcentage de trafic ML.
        
        Args:
            percentage: Entre 0 et 100
        """
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")

        cls._ml_traffic_percentage = percentage
        logger.warning(f"[FeatureFlag] ML traffic set to {percentage}%")

    @classmethod
    def set_fallback_on_error(cls, enabled: bool) -> None:
        """
        Active/désactive le fallback automatique en cas d'erreur.
        
        Args:
            enabled: True pour activer, False pour désactiver
        """
        cls._fallback_on_error = enabled
        logger.warning(
            f"[FeatureFlag] Fallback on error {'enabled' if enabled else 'disabled'}"
        )


def get_feature_flags_status() -> dict[str, Any]:
    """
    Helper pour obtenir le statut complet des feature flags.
    
    Returns:
        Dict avec configuration et stats
    """
    stats = FeatureFlags.get_stats()

    return {
        "config": {
            "ML_ENABLED": FeatureFlags._ml_enabled,
            "ML_TRAFFIC_PERCENTAGE": FeatureFlags._ml_traffic_percentage,
            "FALLBACK_ON_ERROR": FeatureFlags._fallback_on_error,
        },
        "stats": stats,
        "health": {
            "status": "healthy" if stats["ml_success_rate"] > 0.8 else "degraded",
            "success_rate": f"{stats['ml_success_rate']:.1%}",
            "error_rate": f"{1 - stats['ml_success_rate']:.1%}",
        },
    }

