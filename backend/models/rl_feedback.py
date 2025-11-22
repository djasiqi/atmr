# backend/models/rl_feedback.py
"""Modèle pour enregistrer les feedbacks utilisateurs sur les suggestions RL.
Permet l'amélioration continue du modèle DQN via apprentissage supervisé.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from typing_extensions import override

from ext import db


class RLFeedback(db.Model):
    """Feedbacks utilisateurs sur les suggestions RL.

    Permet de :
    - Tracker quelles suggestions ont été appliquées/rejetées
    - Enregistrer les résultats réels (gain, succès)
    - Alimenter le ré-entraînement périodique du modèle DQN
    - Analyser les préférences utilisateurs
    """

    __tablename__ = "rl_feedbacks"

    # Identifiants
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    suggestion_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)

    # Contexte suggestion
    booking_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    assignment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    current_driver_id: Mapped[int] = mapped_column(Integer, nullable=False)
    suggested_driver_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Feedback utilisateur
    # "applied", "rejected", "ignored"
    action: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    # Raison du rejet (optionnel)
    feedback_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Qui a donné le feedback

    # Dates
    created_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    suggestion_generated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )

    # Résultats réels (si appliqué)
    # {gain_minutes, was_better, satisfaction}
    actual_outcome = Column(JSON, nullable=True)
    was_successful: Mapped[bool] = mapped_column(Boolean, nullable=True)
    actual_gain_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Contexte pour ré-entraînement
    # État DQN au moment de la suggestion
    suggestion_state = Column(JSON, nullable=True)
    suggestion_action: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Action DQN choisie
    suggestion_confidence = Column(Float, nullable=True)  # Confiance de la suggestion

    # Métadonnées
    additional_data = Column(JSON, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON."""
        created_at_val = getattr(self, "created_at", None)
        return {
            "id": self.id,
            "company_id": self.company_id,
            "suggestion_id": self.suggestion_id,
            "booking_id": self.booking_id,
            "assignment_id": self.assignment_id,
            "current_driver_id": self.current_driver_id,
            "suggested_driver_id": self.suggested_driver_id,
            "action": self.action,
            "feedback_reason": self.feedback_reason,
            "user_id": self.user_id,
            "created_at": created_at_val.isoformat()
            if created_at_val is not None
            else None,
            "was_successful": self.was_successful,
            "actual_gain_minutes": self.actual_gain_minutes,
            "actual_outcome": self.actual_outcome,
            "suggestion_confidence": self.suggestion_confidence,
        }

    def calculate_reward(self) -> float | None:
        """Calcule la récompense pour le ré-entraînement du modèle.

        Returns:
            Récompense entre -10 et +10
            None si pas encore de résultat

        """
        action_val = getattr(self, "action", None)
        action_str = str(action_val) if action_val is not None else ""

        if action_str == "rejected":
            # Feedback négatif : pénalité modérée
            return -3.0

        if action_str == "ignored":
            # Pas d'action : pénalité légère
            return -1.0

        if action_str == "applied":
            actual_outcome_val = getattr(self, "actual_outcome", None)
            if actual_outcome_val is None:
                # Appliqué mais pas encore de résultat : récompense neutre
                return 0.5

            # Calculer récompense basée sur le résultat
            was_better = actual_outcome_val.get("was_better", False)
            actual_gain_val = getattr(self, "actual_gain_minutes", None)
            gain_value = actual_gain_val if actual_gain_val is not None else 0
            gain = float(gain_value)

            if was_better:
                # Bonne suggestion : récompense proportionnelle au gain
                # Gain de 10 min = +5, Gain de 20 min = +8, etc.
                reward = min(gain / 2.0, 10.0)
                return max(reward, 2.0)  # Au moins +2 si c'était mieux
            # Mauvaise suggestion : pénalité proportionnelle
            penalty = min(abs(gain) / 2.0, 8.0)
            return -max(penalty, 2.0)  # Au moins -2 si c'était pire

        return None

    def is_training_ready(self) -> bool:
        """Vérifie si ce feedback est prêt pour le ré-entraînement.

        Returns:
            True si on a assez d'information pour l'utiliser

        """
        # Besoin au minimum :
        # - État de la suggestion
        # - Action (applied/rejected)
        # - Résultat si appliqué

        suggestion_state_val = getattr(self, "suggestion_state", None)
        if suggestion_state_val is None:
            return False

        action_val = getattr(self, "action", None)
        action_str = str(action_val) if action_val is not None else ""

        if action_str == "rejected":
            return True  # Rejet = feedback négatif direct

        if action_str == "applied":
            actual_outcome_val = getattr(self, "actual_outcome", None)
            return actual_outcome_val is not None  # Besoin du résultat

        return False

    @override
    def __repr__(self):
        return f"<RLFeedback {self.suggestion_id} action={self.action} booking={self.booking_id}>"
