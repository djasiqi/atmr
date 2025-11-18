# backend/models/rl_suggestion_metric.py

# Constantes pour éviter les valeurs magiques
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from typing_extensions import override

from ext import db

EXPECTED_GAIN_MINUTES_ZERO = 0
TOLERANCE_MINUTES = 2

"""Modèle pour tracker les métriques des suggestions RL.
Permet de mesurer la performance du système de suggestions au fil du temps.
"""


class RLSuggestionMetric(db.Model):
    """Métriques de performance pour les suggestions RL.

    Permet de tracker :
    - Quelles suggestions ont été générées
    - Lesquelles ont été appliquées ou rejetées
    - La précision des estimations (gain réel vs estimé)
    - Performance du modèle DQN vs fallback heuristique
    """

    __tablename__ = "rl_suggestion_metrics"

    # Identifiants
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    suggestion_id = Column(String(100), unique=True, nullable=False, index=True)

    # Contexte suggestion
    booking_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    assignment_id: Mapped[int] = mapped_column(Integer, nullable=False)
    current_driver_id: Mapped[int] = mapped_column(Integer, nullable=False)
    suggested_driver_id: Mapped[int] = mapped_column(Integer, nullable=False)

    # Métriques prédites
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    expected_gain_minutes: Mapped[int] = mapped_column(Integer, default=0)
    q_value: Mapped[float] = mapped_column(Float, nullable=True)
    # "dqn_model" ou "basic_heuristic"
    source: Mapped[str] = mapped_column(String(50), nullable=False)

    # Événements
    generated_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    applied_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    rejected_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Résultats réels (si appliqué)
    actual_gain_minutes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    was_successful: Mapped[bool] = mapped_column(Boolean, nullable=True)

    # Données additionnelles (contexte, raisons, etc.)
    additional_data = Column(JSON, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON."""
        generated_at_val = getattr(self, "generated_at", None)
        applied_at_val = getattr(self, "applied_at", None)
        rejected_at_val = getattr(self, "rejected_at", None)
        return {
            "id": self.id,
            "suggestion_id": self.suggestion_id,
            "company_id": self.company_id,
            "booking_id": self.booking_id,
            "assignment_id": self.assignment_id,
            "current_driver_id": self.current_driver_id,
            "suggested_driver_id": self.suggested_driver_id,
            "confidence": self.confidence,
            "expected_gain": self.expected_gain_minutes,
            "actual_gain": self.actual_gain_minutes,
            "gain_accuracy": self.calculate_gain_accuracy(),
            "q_value": self.q_value,
            "source": self.source,
            "applied": applied_at_val is not None,
            "rejected": rejected_at_val is not None,
            "was_successful": self.was_successful,
            "generated_at": generated_at_val.isoformat() if generated_at_val else None,
            "applied_at": applied_at_val.isoformat() if applied_at_val else None,
            "rejected_at": rejected_at_val.isoformat() if rejected_at_val else None,
        }

    def calculate_gain_accuracy(self) -> float | None:
        """Calcule la précision de l'estimation du gain.

        Returns:
            Score entre 0 et 1 (1 = parfait, 0 = très imprécis)
            None si pas encore appliqué

        """
        if self.actual_gain_minutes is None:
            return None

        if self.expected_gain_minutes == EXPECTED_GAIN_MINUTES_ZERO:
            # Si on prédisait 0, on accepte ±2 min comme correct
            return 1.0 if abs(self.actual_gain_minutes) <= TOLERANCE_MINUTES else 0.5

        # Calcul écart relatif
        error = abs(self.actual_gain_minutes - self.expected_gain_minutes)
        relative_error = error / abs(self.expected_gain_minutes)

        # Convertir en score de précision (1 = parfait, 0 = >100% erreur)
        accuracy = max(0.0, 1.0 - relative_error)

        return round(accuracy, 2)

    @override
    def __repr__(self):
        return (
            f"<RLSuggestionMetric {self.suggestion_id} "
            f"booking={self.booking_id} conf={self.confidence} "
            f"source={self.source}>"
        )
