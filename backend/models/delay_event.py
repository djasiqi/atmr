# backend/models/delay_event.py

"""✅ 3.5.1: Modèle pour historique événements retards."""

import logging
from datetime import UTC, datetime
from typing import Any, Dict

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from typing_extensions import override

from ext import db

logger = logging.getLogger(__name__)


class DelayEvent(db.Model):
    """Historique événements retards.

    Stocke les événements de retard détectés pour analytics,
    identification de causes récurrentes et suivi de résolution.
    ON DELETE CASCADE sur assignment_id : à la suppression d'un assignment,
    les delay_events liés sont supprimés (évite ForeignKeyViolation).
    """

    __tablename__ = "delay_events"
    __table_args__ = (
        Index("ix_delay_events_assignment_id", "assignment_id"),
        Index("ix_delay_events_booking_id", "booking_id"),
        Index("ix_delay_events_detected_at", "detected_at"),
        Index("ix_delay_events_severity", "severity"),
        Index("ix_delay_events_resolved_at", "resolved_at"),
    )

    id = Column(Integer, primary_key=True)
    assignment_id = Column(
        Integer,
        ForeignKey("assignment.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    booking_id = Column(Integer, ForeignKey("booking.id"), nullable=False, index=True)

    assignment = relationship(
        "Assignment",
        back_populates="delay_events",
    )

    delay_minutes = Column(
        Integer, nullable=False
    )  # Retard en minutes (peut être négatif = en avance)
    severity = Column(String(20), nullable=False)  # "low", "medium", "high", "critical"
    detected_at = Column(
        DateTime(timezone=True), default=func.now(), nullable=False, index=True
    )
    resolved_at = Column(
        DateTime(timezone=True), nullable=True, index=True
    )  # Null si non résolu

    cause = Column(
        String(100), nullable=True
    )  # "traffic", "driver_late", "booking_delay", etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "id": self.id,
            "assignment_id": self.assignment_id,
            "booking_id": self.booking_id,
            "delay_minutes": self.delay_minutes,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat()
            if bool(getattr(self, "detected_at", None))
            else None,
            "resolved_at": self.resolved_at.isoformat()
            if bool(getattr(self, "resolved_at", None))
            else None,
            "cause": self.cause,
        }

    @override
    def __repr__(self) -> str:
        return (
            f"<DelayEvent id={self.id} assignment_id={self.assignment_id} "
            f"delay={self.delay_minutes}min severity={self.severity}>"
        )

    @staticmethod
    def resolve_delays_for_assignment(
        assignment_id: int, resolved_at: datetime | None = None
    ) -> int:
        """Marque tous les retards non résolus d'un assignment comme résolus.

        Args:
            assignment_id: ID de l'assignment
            resolved_at: Date de résolution (défaut: maintenant)

        Returns:
            Nombre de DelayEvent mis à jour
        """
        if resolved_at is None:
            resolved_at = datetime.now(UTC)

        updated = (
            DelayEvent.query.filter_by(assignment_id=assignment_id)
            .filter(DelayEvent.resolved_at.is_(None))
            .update({"resolved_at": resolved_at}, synchronize_session=False)
        )
        db.session.commit()
        logger.info(
            "Resolved %d delay events for assignment_id=%d", updated, assignment_id
        )
        return updated

    @staticmethod
    def resolve_delays_for_booking(
        booking_id: int, resolved_at: datetime | None = None
    ) -> int:
        """Marque tous les retards non résolus d'un booking comme résolus.

        Args:
            booking_id: ID du booking
            resolved_at: Date de résolution (défaut: maintenant)

        Returns:
            Nombre de DelayEvent mis à jour
        """
        if resolved_at is None:
            resolved_at = datetime.now(UTC)

        updated = (
            DelayEvent.query.filter_by(booking_id=booking_id)
            .filter(DelayEvent.resolved_at.is_(None))
            .update({"resolved_at": resolved_at}, synchronize_session=False)
        )
        db.session.commit()
        logger.info("Resolved %d delay events for booking_id=%d", updated, booking_id)
        return updated
