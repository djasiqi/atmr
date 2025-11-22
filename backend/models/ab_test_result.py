"""Modèle pour stocker les résultats des tests A/B (ML vs Heuristique)."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Float, Integer, String
from typing_extensions import override

from ext import db


class ABTestResult(db.Model):
    """Résultat d'un test A/B comparant ML vs Heuristique."""

    __tablename__ = "ab_test_result"

    id = db.Column(Integer, primary_key=True)
    booking_id = db.Column(
        Integer, db.ForeignKey("booking.id"), nullable=False, index=True
    )
    driver_id = db.Column(
        Integer, db.ForeignKey("driver.id"), nullable=False, index=True
    )
    test_timestamp = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    # Prédiction ML
    ml_delay_minutes = db.Column(Float, nullable=False)
    ml_confidence = db.Column(Float, nullable=True)
    ml_risk_level = db.Column(String(20), nullable=True)
    ml_prediction_time_ms = db.Column(Float, nullable=False)
    ml_weather_factor = db.Column(Float, nullable=True)

    # Prédiction Heuristique
    heuristic_delay_minutes = db.Column(Float, nullable=False)
    heuristic_prediction_time_ms = db.Column(Float, nullable=False)

    # Comparaison
    difference_minutes = db.Column(Float, nullable=False)
    ml_faster = db.Column(db.Boolean, nullable=False)
    speed_advantage_ms = db.Column(Float, nullable=False)

    # Résultat réel (optionnel, rempli plus tard)
    actual_delay_minutes = db.Column(Float, nullable=True)
    ml_error = db.Column(Float, nullable=True)
    heuristic_error = db.Column(Float, nullable=True)
    ml_winner = db.Column(db.Boolean, nullable=True)

    created_at = db.Column(
        db.DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    booking = db.relationship("Booking", backref="ab_test_results", lazy=True)
    driver = db.relationship("Driver", backref="ab_test_results", lazy=True)

    @override
    def __repr__(self) -> str:
        return f"<ABTestResult {self.id} Booking:{self.booking_id} ML:{self.ml_delay_minutes}>"

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "test_timestamp": self.test_timestamp.isoformat()
            if self.test_timestamp
            else None,
            "ml_delay_minutes": round(self.ml_delay_minutes, 2),
            "ml_confidence": round(self.ml_confidence, 3)
            if self.ml_confidence
            else None,
            "ml_risk_level": self.ml_risk_level,
            "ml_prediction_time_ms": round(self.ml_prediction_time_ms, 1),
            "ml_weather_factor": round(self.ml_weather_factor, 2)
            if self.ml_weather_factor
            else None,
            "heuristic_delay_minutes": round(self.heuristic_delay_minutes, 2),
            "heuristic_prediction_time_ms": round(self.heuristic_prediction_time_ms, 1),
            "difference_minutes": round(self.difference_minutes, 2),
            "ml_faster": self.ml_faster,
            "speed_advantage_ms": round(self.speed_advantage_ms, 1),
            "actual_delay_minutes": round(self.actual_delay_minutes, 2)
            if self.actual_delay_minutes is not None
            else None,
            "ml_error": round(self.ml_error, 2) if self.ml_error is not None else None,
            "heuristic_error": round(self.heuristic_error, 2)
            if self.heuristic_error is not None
            else None,
            "ml_winner": self.ml_winner,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def update_actual_result(self, actual_delay_minutes: float) -> None:
        """Met à jour avec le délai réel et calcule les erreurs.

        Args:
            actual_delay_minutes: Délai réel observé

        """
        self.actual_delay_minutes = actual_delay_minutes
        self.ml_error = abs(self.ml_delay_minutes - actual_delay_minutes)
        self.heuristic_error = abs(self.heuristic_delay_minutes - actual_delay_minutes)
        self.ml_winner = self.ml_error < self.heuristic_error
        self.updated_at = datetime.now(timezone.utc)
