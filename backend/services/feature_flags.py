"""Système de feature flags pour contrôler le déploiement ML."""

import logging
from typing import Any, ClassVar, Dict

logger = logging.getLogger(__name__)


class FeatureFlags:
    """Gestionnaire de feature flags pour le système ML."""

    # Configuration par défaut
    _ml_enabled: ClassVar[bool] = False
    _ml_traffic_percentage: ClassVar[int] = 0
    _fallback_on_error: ClassVar[bool] = True

    # Statistiques
    _stats: ClassVar[Dict[str, Any]] = {
        "total_requests": 0,
        "ml_requests": 0,
        "ml_successes": 0,
        "ml_failures": 0,
        "heuristic_requests": 0,
    }

    @classmethod
    def is_ml_enabled(cls, request_id: str | None = None) -> bool:  # noqa: ARG003
        """Vérifie si le ML est activé.

        Args:
            request_id: ID de requête optionnel pour tracking (ignoré actuellement,
                       accepté pour compatibilité avec les tests)
        """
        return cls._ml_enabled

    @classmethod
    def set_ml_enabled(cls, enabled: bool) -> None:
        """Active/désactive le ML."""
        cls._ml_enabled = enabled
        logger.info("ML %s", "activé" if enabled else "désactivé")

    @classmethod
    def get_ml_traffic_percentage(cls) -> int:
        """Retourne le pourcentage de trafic ML."""
        return cls._ml_traffic_percentage

    @classmethod
    def set_ml_traffic_percentage(cls, percentage: int) -> None:
        """Définit le pourcentage de trafic ML."""
        cls._ml_traffic_percentage = max(0, min(100, percentage))
        logger.info("Trafic ML configuré à %d%%", cls._ml_traffic_percentage)

    @classmethod
    def should_fallback_on_error(cls) -> bool:
        """Vérifie si le fallback est activé."""
        return cls._fallback_on_error

    @classmethod
    def set_fallback_on_error(cls, enabled: bool) -> None:
        """Active/désactive le fallback."""
        cls._fallback_on_error = enabled

    @classmethod
    def should_use_ml(cls) -> bool:
        """Détermine si le ML doit être utilisé pour cette requête."""
        if not cls._ml_enabled:
            return False

        # Simulation simple basée sur le pourcentage
        import random

        return random.randint(1, 100) <= cls._ml_traffic_percentage

    @classmethod
    def record_request(cls, used_ml: bool, success: bool = True) -> None:
        """Enregistre une requête dans les statistiques."""
        cls._stats["total_requests"] += 1

        if used_ml:
            cls._stats["ml_requests"] += 1
            if success:
                cls._stats["ml_successes"] += 1
            else:
                cls._stats["ml_failures"] += 1
        else:
            cls._stats["heuristic_requests"] += 1

    @classmethod
    def record_ml_success(cls) -> None:
        """Enregistre un succès ML (convenience method)."""
        cls.record_request(used_ml=True, success=True)

    @classmethod
    def record_ml_failure(cls) -> None:
        """Enregistre un échec ML (convenience method)."""
        cls.record_request(used_ml=True, success=False)

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """Retourne les statistiques."""
        stats = cls._stats.copy()

        # Ajouter la configuration actuelle
        stats["ml_enabled"] = cls._ml_enabled
        stats["ml_traffic_percentage"] = cls._ml_traffic_percentage

        # Calculer le taux de succès ML
        if stats["ml_requests"] > 0:
            stats["ml_success_rate"] = stats["ml_successes"] / stats["ml_requests"]
        else:
            stats["ml_success_rate"] = 0.0

        return stats

    @classmethod
    def reset_stats(cls) -> None:
        """Réinitialise les statistiques."""
        cls._stats = {
            "total_requests": 0,
            "ml_requests": 0,
            "ml_successes": 0,
            "ml_failures": 0,
            "heuristic_requests": 0,
        }
        logger.info("Statistiques réinitialisées")


def get_feature_flags_status() -> Dict[str, Any]:
    """Retourne le statut complet des feature flags."""
    return {
        "config": {
            "ML_ENABLED": FeatureFlags.is_ml_enabled(),
            "ML_TRAFFIC_PERCENTAGE": FeatureFlags.get_ml_traffic_percentage(),
            "FALLBACK_ON_ERROR": FeatureFlags.should_fallback_on_error(),
        },
        "stats": FeatureFlags.get_stats(),
    }
