# backend/models/eta_accuracy_log.py

"""Modèle pour tracking précision ETA (prédit vs réel).

Permet de :
- Logger toutes les prédictions ETA
- Comparer prédictions vs réalité après trajet
- Calculer métriques précision (MAE, RMSE, etc.)
- Identifier zones/heures problématiques
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Float, ForeignKey, Integer, String
from typing_extensions import override

from ext import db


class EtaAccuracyLog(db.Model):
    """Log précision ETA pour analytics.

    Stocke chaque prédiction ETA et permet de la comparer
    avec la durée réelle une fois le trajet terminé.
    """

    __tablename__ = "eta_accuracy_log"

    # Clé primaire
    id = db.Column(Integer, primary_key=True)

    # Identifiants
    booking_id = db.Column(Integer, ForeignKey("booking.id"), nullable=True, index=True)
    assignment_id = db.Column(
        Integer, ForeignKey("assignment.id"), nullable=True, index=True
    )

    # Prédiction ETA
    predicted_eta_seconds = db.Column(Integer, nullable=False)
    actual_duration_seconds = db.Column(Integer, nullable=True)  # Rempli après trajet
    error_seconds = db.Column(Integer, nullable=True)  # Différence (actual - predicted)

    # Coordonnées
    origin_lat = db.Column(Float, nullable=False)
    origin_lon = db.Column(Float, nullable=False)
    dest_lat = db.Column(Float, nullable=False)
    dest_lon = db.Column(Float, nullable=False)

    # Source ETA
    source = db.Column(
        String(50), nullable=False
    )  # "osrm", "osrm_ml", "haversine", etc.
    ml_confidence = db.Column(Float, nullable=True)  # Confiance ML (0.0-1.0)

    # Timestamps
    created_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
        index=True,
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )

    # Relations
    booking = db.relationship(
        "Booking", backref=db.backref("eta_accuracy_logs", lazy="dynamic")
    )
    assignment = db.relationship(
        "Assignment", backref=db.backref("eta_accuracy_logs", lazy="dynamic")
    )

    @override
    def __repr__(self) -> str:
        return (
            f"<EtaAccuracyLog booking_id={self.booking_id} "
            f"predicted={self.predicted_eta_seconds}s "
            f"actual={self.actual_duration_seconds or 'N/A'}s "
            f"source={self.source}>"
        )

    def update_actual_duration(self, actual_duration_seconds: int) -> None:
        """Met à jour la durée réelle et calcule l'erreur.

        Args:
            actual_duration_seconds: Durée réelle en secondes
        """
        self.actual_duration_seconds = actual_duration_seconds
        self.error_seconds = actual_duration_seconds - self.predicted_eta_seconds
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour API."""
        return {
            "id": self.id,
            "booking_id": self.booking_id,
            "assignment_id": self.assignment_id,
            "predicted_eta_seconds": self.predicted_eta_seconds,
            "actual_duration_seconds": self.actual_duration_seconds,
            "error_seconds": self.error_seconds,
            "origin": {"lat": self.origin_lat, "lon": self.origin_lon},
            "destination": {"lat": self.dest_lat, "lon": self.dest_lon},
            "source": self.source,
            "ml_confidence": self.ml_confidence,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
