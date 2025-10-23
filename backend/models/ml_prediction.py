"""
Modèle pour stocker les prédictions ML et leurs résultats réels.

Permet de :
- Tracker toutes les prédictions ML
- Comparer prédictions vs réalité
- Calculer métriques (MAE, R²)
- Détecter drift
"""
from datetime import datetime

from sqlalchemy import Float, Integer, String, Text

from ext import db


class MLPrediction(db.Model):
    """
    Prédiction ML avec résultat réel pour monitoring.
    Stocke chaque prédiction ML et permet de la comparer
    avec le retard réel une fois la course terminée.
    """
    __tablename__ = "ml_prediction"

    # Clé primaire
    id = db.Column(Integer, primary_key=True)

    # Identifiants
    booking_id = db.Column(Integer, db.ForeignKey("booking.id"), nullable=False, index=True)
    driver_id = db.Column(Integer, db.ForeignKey("driver.id"), nullable=True, index=True)
    request_id = db.Column(String(100), nullable=True, index=True)

    # Prédiction ML
    predicted_delay_minutes = db.Column(Float, nullable=False)
    confidence = db.Column(Float, nullable=False)  # 0.0 - 1.0
    risk_level = db.Column(String(20), nullable=False)  # low, medium, high
    contributing_factors = db.Column(Text, nullable=True)  # JSON string

    # Contexte prédiction
    model_version = db.Column(String(50), nullable=True)
    prediction_time_ms = db.Column(Float, nullable=True)  # Temps de calcul
    feature_flag_enabled = db.Column(db.Boolean, default=True)
    traffic_percentage = db.Column(Integer, nullable=True)

    # Résultat réel (rempli après course)
    actual_delay_minutes = db.Column(Float, nullable=True)
    actual_pickup_at = db.Column(db.DateTime, nullable=True)
    actual_dropoff_at = db.Column(db.DateTime, nullable=True)

    # Métriques calculées
    prediction_error = db.Column(Float, nullable=True)  # |predicted - actual|
    is_accurate = db.Column(db.Boolean, nullable=True)  # error < 3 min

    # Métadonnées
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)  # noqa: DTZ003
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)  # noqa: DTZ003

    # Relations
    booking = db.relationship("Booking", backref=db.backref("ml_predictions", lazy="dynamic"))
    driver = db.relationship("Driver", backref=db.backref("ml_predictions", lazy="dynamic"))

    def __repr__(self) -> str:
        return (
            f"<MLPrediction booking_id={self.booking_id} "
            f"predicted={self.predicted_delay_minutes:.2f}min "
            f"actual={self.actual_delay_minutes or 'N/A'}>"
        )

    def update_actual_delay(self, actual_delay: float) -> None:
        """
        Met à jour le retard réel et calcule les métriques.
        Args:
            actual_delay: Retard réel en minutes
        """
        self.actual_delay_minutes = actual_delay
        self.prediction_error = abs(self.predicted_delay_minutes - actual_delay)
        self.is_accurate = self.prediction_error < 3.0  # Seuil: 3 min

    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour API."""
        return {
            "id": self.id,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "request_id": self.request_id,
            "predicted_delay_minutes": self.predicted_delay_minutes,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "contributing_factors": self.contributing_factors,
            "model_version": self.model_version,
            "prediction_time_ms": self.prediction_time_ms,
            "actual_delay_minutes": self.actual_delay_minutes,
            "prediction_error": self.prediction_error,
            "is_accurate": self.is_accurate,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

