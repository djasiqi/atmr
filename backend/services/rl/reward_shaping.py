#!/usr/bin/env python3
"""
Module de reward shaping avancé pour l'environnement de dispatch.

Ce module implémente un système de récompenses sophistiqué qui guide l'agent
vers des comportements optimaux en termes de:
- Ponctualité (ALLER vs RETOUR)
- Efficacité distance
- Équité de charge
- Satisfaction client

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class AdvancedRewardShaping:
    """
    Système de reward shaping avancé avec poids configurables.
    
    Features:
    - Fonctions piecewise pour ponctualité
    - Log-scaling pour distances
    - Bonus équité de charge
    - Pénalités progressives
    - Configuration via settings
    """

    def __init__(
        self,
        punctuality_weight: float = 1.0,
        distance_weight: float = 0.5,
        equity_weight: float = 0.3,
        efficiency_weight: float = 0.2,
        satisfaction_weight: float = 0.4,
    ):
        """
        Initialise le système de reward shaping.

        Args:
            punctuality_weight: Poids pour la ponctualité (1.0 = priorité max)
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

        # Seuils configurables
        self.aller_tolerance = 0.0  # ALLER: 0 tolérance
        self.retour_tolerance_soft = 15.0  # RETOUR: tolérance douce 15min
        self.retour_tolerance_hard = 30.0  # RETOUR: tolérance dure 30min
        self.short_distance_threshold = 5.0  # Distance courte < 5km
        self.max_distance_penalty = 50.0  # Pénalité max distance
        self.excellent_equity_threshold = 1.0  # Écart ≤ 1 course
        self.good_equity_threshold = 2.0  # Écart ≤ 2 courses

        logger.info(
            "[RewardShaping] Initialisé avec poids: punct=%.1f, dist=%.1f, equity=%.1f, eff=%.1f, sat=%.1f",
            punctuality_weight, distance_weight, equity_weight, efficiency_weight, satisfaction_weight
        )

    def calculate_reward(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        info: Dict[str, Any]
    ) -> float:
        """
        Calcule la récompense totale avec shaping avancé.

        Args:
            state: État actuel
            action: Action prise
            next_state: État suivant
            info: Informations additionnelles

        Returns:
            Récompense totale calculée
        """
        reward = 0.0

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
        """
        Calcule la récompense basée sur la ponctualité avec fonction piecewise.

        Args:
            info: Informations sur l'assignation

        Returns:
            Récompense de ponctualité
        """
        if not info.get('is_late', False):
            # Ponctualité parfaite
            lateness = info.get('lateness_minutes', 0)
            if lateness <= 0:
                return 100.0  # Bonus parfait
            else:
                # Bonus décroissant avec l'avance
                return max(50.0, 100.0 - lateness * 2.0)

        # Retard détecté
        lateness = info.get('lateness_minutes', 0)
        is_outbound = info.get('is_outbound', False)

        if is_outbound:  # ALLER: 0 tolérance
            return -min(200.0, lateness * 10.0)
        else:  # RETOUR: tolérance progressive
            if lateness <= self.retour_tolerance_soft:
                return 0.0  # Neutre dans la tolérance douce
            elif lateness <= self.retour_tolerance_hard:
                # Pénalité progressive
                return -(lateness - self.retour_tolerance_soft) * 2.0
            else:
                # Pénalité forte
                return -min(100.0, lateness * 3.0)

    def _calculate_distance_reward(self, info: Dict[str, Any]) -> float:
        """
        Calcule la récompense basée sur la distance avec log-scaling.

        Args:
            info: Informations sur l'assignation

        Returns:
            Récompense de distance
        """
        distance = info.get('distance_km', 0)

        if distance < self.short_distance_threshold:
            # Bonus pour distance courte
            return 20.0 + (self.short_distance_threshold - distance) * 4.0
        else:
            # Log penalty pour distances longues
            return max(-self.max_distance_penalty, -np.log(distance) * 10.0)

    def _calculate_equity_reward(self, info: Dict[str, Any]) -> float:
        """
        Calcule la récompense basée sur l'équité de charge.

        Args:
            info: Informations sur les charges des chauffeurs

        Returns:
            Récompense d'équité
        """
        loads = info.get('driver_loads', [])
        if not loads or len(loads) < 2:
            return 0.0

        load_std = np.std(loads)

        if load_std < self.excellent_equity_threshold:
            return 100.0  # Excellent équilibre
        elif load_std < self.good_equity_threshold:
            return 50.0   # Bon équilibre
        else:
            return -load_std * 10.0  # Pénalité déséquilibre

    def _calculate_efficiency_reward(self, info: Dict[str, Any]) -> float:
        """
        Calcule la récompense basée sur l'efficacité système.

        Args:
            info: Informations sur l'efficacité

        Returns:
            Récompense d'efficacité
        """
        # Bonus pour assignation réussie
        if info.get('assignment_successful', False):
            base_reward = 50.0

            # Bonus pour assignation rapide
            assignment_time = info.get('assignment_time_minutes', 0)
            if assignment_time < 5:
                base_reward += 20.0
            elif assignment_time < 10:
                base_reward += 10.0

            return base_reward

        # Pénalité pour échec d'assignation
        return -20.0

    def _calculate_satisfaction_reward(self, info: Dict[str, Any]) -> float:
        """
        Calcule la récompense basée sur la satisfaction client.

        Args:
            info: Informations sur la satisfaction

        Returns:
            Récompense de satisfaction
        """
        # Bonus pour chauffeur REGULAR (préféré)
        if info.get('driver_type') == 'REGULAR':
            return 20.0

        # Bonus pour respect des préférences
        if info.get('respects_preferences', False):
            return 15.0

        # Pénalité pour chauffeur EMERGENCY utilisé pour course normale
        if info.get('driver_type') == 'EMERGENCY' and info.get('booking_priority', 3) < 4:
            return -10.0

        return 0.0

    def update_weights(self, **kwargs) -> None:
        """
        Met à jour les poids de récompense dynamiquement.

        Args:
            **kwargs: Nouveaux poids (punctuality_weight, distance_weight, etc.)
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                logger.info("[RewardShaping] Poids %s mis à jour: %.2f", key, value)

    def get_current_weights(self) -> Dict[str, float]:
        """
        Retourne les poids actuels.

        Returns:
            Dictionnaire des poids actuels
        """
        return {
            'punctuality_weight': self.punctuality_weight,
            'distance_weight': self.distance_weight,
            'equity_weight': self.equity_weight,
            'efficiency_weight': self.efficiency_weight,
            'satisfaction_weight': self.satisfaction_weight,
        }

    def reset(self) -> None:
        """Remet à zéro les statistiques internes."""
        logger.debug("[RewardShaping] Reset des statistiques")


class RewardShapingConfig:
    """
    Configuration centralisée pour le reward shaping.
    
    Permet de définir différents profils de récompenses selon le contexte.
    """

    # Profil par défaut (équilibré)
    DEFAULT = {
        'punctuality_weight': 1.0,
        'distance_weight': 0.5,
        'equity_weight': 0.3,
        'efficiency_weight': 0.2,
        'satisfaction_weight': 0.4,
    }

    # Profil ponctualité (priorité aux retards)
    PUNCTUALITY_FOCUSED = {
        'punctuality_weight': 1.5,
        'distance_weight': 0.3,
        'equity_weight': 0.2,
        'efficiency_weight': 0.1,
        'satisfaction_weight': 0.3,
    }

    # Profil équité (priorité à l'équilibre)
    EQUITY_FOCUSED = {
        'punctuality_weight': 0.8,
        'distance_weight': 0.4,
        'equity_weight': 0.6,
        'efficiency_weight': 0.2,
        'satisfaction_weight': 0.3,
    }

    # Profil efficacité (priorité aux distances)
    EFFICIENCY_FOCUSED = {
        'punctuality_weight': 0.7,
        'distance_weight': 1.0,
        'equity_weight': 0.2,
        'efficiency_weight': 0.4,
        'satisfaction_weight': 0.2,
    }

    @classmethod
    def get_profile(cls, profile_name: str) -> Dict[str, float]:
        """
        Retourne un profil de configuration.

        Args:
            profile_name: Nom du profil ('DEFAULT', 'PUNCTUALITY_FOCUSED', etc.)

        Returns:
            Dictionnaire de configuration
        """
        return getattr(cls, profile_name.upper(), cls.DEFAULT)
