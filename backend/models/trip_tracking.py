# backend/models/trip_tracking.py

"""✅ 3.3.3: Modèle pour historique positions pendant trajet."""

import logging
from typing import Any, Dict

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer
from sqlalchemy.sql import func
from typing_extensions import override

from ext import db

logger = logging.getLogger(__name__)


class TripTracking(db.Model):
    """Historique positions pendant trajet.

    Stocke les positions GPS du chauffeur pendant un trajet (assignment)
    pour permettre replay, analytics et debugging.
    """

    __tablename__ = "trip_tracking"
    __table_args__ = (
        Index("ix_trip_tracking_assignment_timestamp", "assignment_id", "timestamp"),
        Index("ix_trip_tracking_booking_id", "booking_id"),
        Index("ix_trip_tracking_driver_id", "driver_id"),
        Index("ix_trip_tracking_timestamp", "timestamp"),
    )

    id = Column(Integer, primary_key=True)
    assignment_id = Column(
        Integer, ForeignKey("assignment.id"), nullable=False, index=True
    )
    booking_id = Column(Integer, ForeignKey("booking.id"), nullable=False, index=True)
    driver_id = Column(Integer, ForeignKey("driver.id"), nullable=False, index=True)

    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    speed = Column(Float, nullable=True)  # m/s
    heading = Column(Float, nullable=True)  # degrés (0-360)
    accuracy = Column(Float, nullable=True)  # mètres

    timestamp = Column(
        DateTime(timezone=True), default=func.now(), nullable=False, index=True
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "assignment_id": self.assignment_id,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "speed": self.speed,
            "heading": self.heading,
            "accuracy": self.accuracy,
            "timestamp": self.timestamp.isoformat() if bool(self.timestamp) else None,
        }

    @override
    def __repr__(self) -> str:
        return (
            f"<TripTracking id={self.id} assignment_id={self.assignment_id} "
            f"driver_id={self.driver_id} lat={self.latitude} lon={self.longitude}>"
        )
