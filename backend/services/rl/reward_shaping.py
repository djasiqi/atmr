
# Constantes pour éviter les valeurs magiques
import logging
from typing import Any, ClassVar, Dict

import numpy as np

COURTE_THRESHOLD = 5
MIN_DRIVERS_FOR_EQUITY = 2
HIGH_PRIORITY_THRESHOLD = 4
LATENESS_ZERO = 0
ASSIGNMENT_TIME_THRESHOLD = 5

"""Module de reward shaping avancé pour l'environnement de dispatch.

Ce module implémente un système de récompenses sophistiqué qui guide l'agent
vers des comportements optimaux en termes de:
- Ponctualité (ALLER vs RETOUR)
- Efficacité distance
- Équité de charge
- Satisfaction client

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""


logger = logging.getLogger(__name__)


class AdvancedRewardShaping:
    """Système de reward shaping avancé avec poids configurables.

    Features:
    - Fonctions piecewise pour ponctualité
    - Log-scaling pour distances
    - Bonus équité de charge
    - Pénalités progressives
    - Configuration via settings
    """
    
    # Déclaration de variable d'instance pour le linter
    _custom_weights: Dict[str, float] | None = None

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self,
        punctuality_weight: float = 1,
        distance_weight: float = 0.5,
        equity_weight: float = 0.3,
        efficiency_weight: float = 0.2,
        satisfaction_weight: float = 0.4,
    ):
        """Initialise le système de reward shaping.

        Args:
            punctuality_weight: Poids pour la ponctualité (1 = priorité max)
            distance_weight: Poids pour l'efficacité distance (0.5)
            equity_weight: Poids pour l'équité de charge (0.3)
            efficiency_weight: Poids pour l'efficacité système (0.2)
            satisfaction_weight: Poids pour la satisfaction client (0.4)

        """
        self.punctuality_weight = punctuality_weight
        self.distance_weight = distance_weight
        self.equity_weight = equity_weight
        self.efficiency_weight = efficiency_weight
        self.satisfaction_weight = satisfaction_weight

        # Variable d'instance pour les poids personnalisés
        self._custom_weights = None

        # Seuils configurables
        self.aller_tolerance = 0  # ALLER: 0 tolérance
        self.retour_tolerance_soft = 15  # RETOUR: tolérance douce 15min
        self.retour_tolerance_hard = 30  # RETOUR: tolérance dure 30min
        # Distance courte < COURTE_THRESHOLDkm
        self.short_distance_threshold = COURTE_THRESHOLD
        self.max_distance_penalty = 50  # Pénalité max distance
        self.excellent_equity_threshold = 1  # Écart ≤ 1 course
        self.good_equity_threshold = 2  # Écart ≤ 2 courses

        logger.info(
            "[RewardShaping] Initialisé avec poids: punct=%.1f, dist=%.1f, equity=%.1f, eff=%.1f, sat=%.1f",
            punctuality_weight, distance_weight, equity_weight, efficiency_weight, satisfaction_weight
        )

    def calculate_reward(
        self,
        state: np.ndarray[Any, np.dtype[np.float32]],  # noqa: ARG002
        action: int,  # noqa: ARG002
        next_state: np.ndarray[Any, np.dtype[np.float32]],  # noqa: ARG002
        info: Dict[str, Any]
    ) -> float:
        """Calcule la récompense totale avec shaping avancé.

        Args:
            state: État actuel
            action: Action prise
            next_state: État suivant
            info: Informations additionnelles

        Returns:
            Récompense totale calculée

        """
        reward = 0

        # 1. Punctuality reward (piecewise)
        punctuality_reward = self._calculate_punctuality_reward(info)
        reward += self.punctuality_weight * punctuality_reward

        # 2. Distance efficiency (log-scaled)
        distance_reward = self._calculate_distance_reward(info)
        reward += self.distance_weight * distance_reward

        # 3. Workload equity
        equity_reward = self._calculate_equity_reward(info)
        reward += self.equity_weight * equity_reward

        # 4. System efficiency
        efficiency_reward = self._calculate_efficiency_reward(info)
        reward += self.efficiency_weight * efficiency_reward

        # 5. Client satisfaction
        satisfaction_reward = self._calculate_satisfaction_reward(info)
        reward += self.satisfaction_weight * satisfaction_reward

        # Logging détaillé pour debug
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[RewardShaping] Détail: punct=%.1f, dist=%.1f, equity=%.1f, eff=%.1f, sat=%.1f → total=%.1f",
                punctuality_reward, distance_reward, equity_reward, efficiency_reward, satisfaction_reward, reward
            )

        return reward

    def _calculate_punctuality_reward(self, info: Dict[str, Any]) -> float:
        """Calcule la récompense basée sur la ponctualité avec fonction piecewise.

        Args:
            info: Informations sur l'assignation

        Returns:
            Récompense de ponctualité

        """
        if not info.get("is_late", False):
            # Ponctualité parfaite
            lateness = info.get("lateness_minutes", 0)
            if lateness <= LATENESS_ZERO:
                return 100  # Bonus parfait
            # Bonus décroissant avec l'avance
            return max(50, 100 - lateness * 2)

        # Retard détecté
        lateness = info.get("lateness_minutes", 0)
        is_outbound = info.get("is_outbound", False)

        if is_outbound:  # ALLER: 0 tolérance
            return -min(200, lateness * 10)
        # RETOUR: tolérance progressive
        if lateness <= self.retour_tolerance_soft:
            return 0  # Neutre dans la tolérance douce
        if lateness <= self.retour_tolerance_hard:
            # Pénalité progressive
            return -(lateness - self.retour_tolerance_soft) * 2
        # Pénalité forte
        return -min(100, lateness * 3)

    def _calculate_distance_reward(self, info: Dict[str, Any]) -> float:
        """Calcule la récompense basée sur la distance avec log-scaling.

        Args:
            info: Informations sur l'assignation

        Returns:
            Récompense de distance

        """
        distance = info.get("distance_km", 0)

        if distance < self.short_distance_threshold:
            # Bonus pour distance courte
            return 20 + (self.short_distance_threshold - distance) * 4
        # Log penalty pour distances longues
        return max(-self.max_distance_penalty, -np.log(distance) * 10)

    def _calculate_equity_reward(self, info: Dict[str, Any]) -> float:
        """Calcule la récompense basée sur l'équité de charge.

        Args:
            info: Informations sur les charges des chauffeurs

        Returns:
            Récompense d'équité

        """
        loads = info.get("driver_loads", [])
        if not loads or len(loads) < MIN_DRIVERS_FOR_EQUITY:
            return 0

        load_std = np.std(loads)

        if load_std < self.excellent_equity_threshold:
            return 100  # Excellent équilibre
        if load_std < self.good_equity_threshold:
            return 50   # Bon équilibre
        return float(-load_std * 10)  # Pénalité déséquilibre

    def _calculate_efficiency_reward(self, info: Dict[str, Any]) -> float:
        """Calcule la récompense basée sur l'efficacité système.

        Args:
            info: Informations sur l'efficacité

        Returns:
            Récompense d'efficacité

        """
        # Bonus pour assignation réussie
        if info.get("assignment_successful", False):
            base_reward = 50

            # Bonus pour assignation rapide
            assignment_time = info.get("assignment_time_minutes", 0)
            if assignment_time < ASSIGNMENT_TIME_THRESHOLD:
                base_reward += 20
            elif assignment_time < ASSIGNMENT_TIME_THRESHOLD:
                base_reward += 10

            return base_reward

        # Pénalité pour échec d'assignation
        return -20

    def _calculate_satisfaction_reward(self, info: Dict[str, Any]) -> float:
        """Calcule la récompense basée sur la satisfaction client.

        Args:
            info: Informations sur la satisfaction

        Returns:
            Récompense de satisfaction

        """
        # Bonus pour chauffeur REGULAR (préféré)
        if info.get("driver_type") == "REGULAR":
            return 20

        # Bonus pour respect des préférences
        if info.get("respects_preferences", False):
            return 15

        # Pénalité pour chauffeur EMERGENCY utilisé pour course normale
        if info.get("driver_type") == "EMERGENCY" and info.get(
                "booking_priority", 3) < HIGH_PRIORITY_THRESHOLD:
            return -10

        return 0

    def update_weights(self, **kwargs) -> None:
        """Met à jour les poids de récompense dynamiquement.

        Args:
            **kwargs: Nouveaux poids (punctuality_weight, distance_weight, etc.)

        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info(
                    "[RewardShaping] Poids %s mis à jour: %.2f", key, value)

    def get_current_weights(self) -> Dict[str, float]:
        """Retourne les poids actuels.

        Returns:
            Dictionnaire des poids actuels

        """
        return {
            "punctuality_weight": self.punctuality_weight,
            "distance_weight": self.distance_weight,
            "equity_weight": self.equity_weight,
            "efficiency_weight": self.efficiency_weight,
            "satisfaction_weight": self.satisfaction_weight,
        }

    def reset(self) -> None:
        """Remet à zéro les statistiques internes."""
        logger.debug("[RewardShaping] Reset des statistiques")


class RewardShapingConfig:
    """Configuration centralisée pour le reward shaping.

    Permet de définir différents profils de récompenses selon le contexte.
    """

    # Profil par défaut (équilibré)
    DEFAULT: ClassVar[Dict[str, float]] = {
        "punctuality_weight": 1,
        "distance_weight": 0.5,
        "equity_weight": 0.3,
        "efficiency_weight": 0.2,
        "satisfaction_weight": 0.4,
    }

    # Profil ponctualité (priorité aux retards)
    PUNCTUALITY_FOCUSED: ClassVar[Dict[str, float]] = {
        "punctuality_weight": 1.5,
        "distance_weight": 0.3,
        "equity_weight": 0.2,
        "efficiency_weight": 0.1,
        "satisfaction_weight": 0.3,
    }

    # Profil équité (priorité à l'équilibre)
    EQUITY_FOCUSED: ClassVar[Dict[str, float]] = {
        "punctuality_weight": 0.8,
        "distance_weight": 0.4,
        "equity_weight": 0.6,
        "efficiency_weight": 0.2,
        "satisfaction_weight": 0.3,
    }

    # Profil efficacité (priorité aux distances)
    EFFICIENCY_FOCUSED: ClassVar[Dict[str, float]] = {
        "punctuality_weight": 0.7,
        "distance_weight": 1,
        "equity_weight": 0.2,
        "efficiency_weight": 0.4,
        "satisfaction_weight": 0.2,
    }

    @classmethod
    def get_profile(cls, profile_name: str) -> Dict[str, float]:
        """Retourne un profil de configuration.

        Args:
            profile_name: Nom du profil ('DEFAULT', 'PUNCTUALITY_FOCUSED', etc.)

        Returns:
            Dictionnaire de configuration

        """
        return getattr(cls, profile_name.upper(), cls.DEFAULT)

    def get_weights(self) -> Dict[str, float]:
        """Retourne les poids de configuration.

        Returns:
            Dictionnaire des poids

        """
        if hasattr(self, "_custom_weights") and self._custom_weights:
            return self._custom_weights
        return self.DEFAULT

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Met à jour les poids de configuration.

        Args:
            weights: Nouveaux poids

        """
        # Cette méthode pourrait être implémentée pour modifier les profils
        # Pour l'instant, on ne fait rien car les profils sont statiques
        # Mais on peut stocker les poids dans une variable d'instance
        self._custom_weights = weights.copy()  # type: ignore
