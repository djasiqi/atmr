# backend/models/rl_suggestion_metric.py
"""
Modèle pour tracker les métriques des suggestions RL.
Permet de mesurer la performance du système de suggestions au fil du temps.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Column, DateTime, Float, Integer, String, Boolean, JSON
from sqlalchemy.sql import func

from ext import db


class RLSuggestionMetric(db.Model):
    """
    Métriques de performance pour les suggestions RL.
    
    Permet de tracker :
    - Quelles suggestions ont été générées
    - Lesquelles ont été appliquées ou rejetées
    - La précision des estimations (gain réel vs estimé)
    - Performance du modèle DQN vs fallback heuristique
    """
    __tablename__ = 'rl_suggestion_metrics'

    # Identifiants
    id = Column(Integer, primary_key=True)
    company_id = Column(Integer, nullable=False, index=True)
    suggestion_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Contexte suggestion
    booking_id = Column(Integer, nullable=False, index=True)
    assignment_id = Column(Integer, nullable=False)
    current_driver_id = Column(Integer, nullable=False)
    suggested_driver_id = Column(Integer, nullable=False)
    
    # Métriques prédites
    confidence = Column(Float, nullable=False)
    expected_gain_minutes = Column(Integer, default=0)
    q_value = Column(Float, nullable=True)
    source = Column(String(50), nullable=False)  # "dqn_model" ou "basic_heuristic"
    
    # Événements
    generated_at = Column(DateTime, nullable=False, default=func.now(), index=True)
    applied_at = Column(DateTime, nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    
    # Résultats réels (si appliqué)
    actual_gain_minutes = Column(Integer, nullable=True)
    was_successful = Column(Boolean, nullable=True)
    
    # Données additionnelles (contexte, raisons, etc.)
    additional_data = Column(JSON, nullable=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON."""
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
            "applied": self.applied_at is not None,
            "rejected": self.rejected_at is not None,
            "was_successful": self.was_successful,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "rejected_at": self.rejected_at.isoformat() if self.rejected_at else None,
        }
    
    def calculate_gain_accuracy(self) -> float | None:
        """
        Calcule la précision de l'estimation du gain.
        
        Returns:
            Score entre 0 et 1 (1 = parfait, 0 = très imprécis)
            None si pas encore appliqué
        """
        if self.actual_gain_minutes is None:
            return None
        
        if self.expected_gain_minutes == 0:
            # Si on prédisait 0, on accepte ±2 min comme correct
            return 1.0 if abs(self.actual_gain_minutes) <= 2 else 0.5
        
        # Calcul écart relatif
        error = abs(self.actual_gain_minutes - self.expected_gain_minutes)
        relative_error = error / abs(self.expected_gain_minutes)
        
        # Convertir en score de précision (1 = parfait, 0 = >100% erreur)
        accuracy = max(0.0, 1.0 - relative_error)
        
        return round(accuracy, 2)
    
    def __repr__(self):
        return (
            f"<RLSuggestionMetric {self.suggestion_id} "
            f"booking={self.booking_id} conf={self.confidence:.2f} "
            f"source={self.source}>"
        )

