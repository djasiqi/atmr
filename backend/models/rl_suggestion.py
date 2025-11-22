# backend/models/rl_suggestion.py
"""Modèle pour les suggestions RL en mode shadow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from sqlalchemy import JSON, TIMESTAMP, Column, Float, ForeignKey, Integer
from sqlalchemy.orm import relationship

from ext import db


class RLSuggestion(db.Model):
    """Suggestion générée par le RL en mode shadow.

    Stocke les suggestions RL pour comparaison avec les heuristiques,
    sans impact sur la production.
    """

    __tablename__ = "rl_suggestions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dispatch_run_id = Column(
        Integer, ForeignKey("dispatch_run.id"), nullable=False, index=True
    )
    booking_id = Column(Integer, ForeignKey("booking.id"), nullable=False, index=True)
    driver_id = Column(Integer, ForeignKey("driver.id"), nullable=False, index=True)
    score = Column(Float, nullable=False)
    kpi_snapshot = Column(
        JSON, nullable=True, comment="Snapshot des KPIs au moment de la suggestion"
    )
    created_at = Column(
        TIMESTAMP(timezone=True), nullable=False, default=datetime.utcnow, index=True
    )

    # Relations
    dispatch_run = relationship("DispatchRun", backref="rl_suggestions")
    booking = relationship("Booking", backref="rl_suggestions")
    driver = relationship("Driver", backref="rl_suggestions")

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            "id": self.id,
            "dispatch_run_id": self.dispatch_run_id,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "score": self.score,
            "kpi_snapshot": self.kpi_snapshot,
            "created_at": self.created_at.isoformat(),
        }

    def __repr__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        return f"<RLSuggestion(id={self.id}, booking_id={self.booking_id}, driver_id={self.driver_id}, score={self.score})>"
