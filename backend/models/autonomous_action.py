"""Modèle pour tracer toutes les actions automatiques du système.

Ce modèle est essentiel pour :
- Audit trail complet de toutes les actions autonomes
- Rate limiting (limiter le nombre d'actions par heure/jour)
- Monitoring et détection d'anomalies
- Compliance et traçabilité
"""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Float, Integer, String, Text
from typing_extensions import override

from ext import db


class AutonomousAction(db.Model):
    """Trace chaque action automatique effectuée par le système.

    Utilisé pour :
    - Audit trail complet
    - Rate limiting (éviter trop d'actions)
    - Monitoring des performances
    - Rollback si nécessaire
    """

    __tablename__ = "autonomous_action"

    # Clé primaire
    id = db.Column(Integer, primary_key=True)

    # Identifiants
    company_id = db.Column(
        Integer, db.ForeignKey("company.id"), nullable=False, index=True
    )
    booking_id = db.Column(
        Integer, db.ForeignKey("booking.id"), nullable=True, index=True
    )
    driver_id = db.Column(
        Integer, db.ForeignKey("driver.id"), nullable=True, index=True
    )

    # Type d'action
    action_type = db.Column(String(50), nullable=False, index=True)
    """
    Types possibles:
    - 'reassign': Réassignation chauffeur
    - 'adjust_time': Ajustement temps pickup/dropoff
    - 'notify_customer': Notification client automatique
    - 'redistribute': Redistribution des charges
    - 'reoptimize': Ré-optimisation globale
    """

    # Détails de l'action
    action_description = db.Column(String(500), nullable=False)
    # JSON avec détails (avant/après)
    action_data = db.Column(Text, nullable=True)

    # Résultat
    success = db.Column(Boolean, nullable=False, default=True)
    error_message = db.Column(Text, nullable=True)

    # Métriques
    execution_time_ms = db.Column(Float, nullable=True)  # Temps d'exécution
    confidence_score = db.Column(Float, nullable=True)  # Confiance ML (0.0-1.0)
    expected_improvement_minutes = db.Column(Float, nullable=True)  # Gain attendu

    # Contexte
    trigger_source = db.Column(String(100), nullable=True)
    """
    Sources possibles:
    - 'autorun_scheduler': Dispatch automatique périodique
    - 'realtime_optimizer': Optimisation temps réel
    - 'ml_prediction': Déclenchée par prédiction ML
    - 'manual_trigger': Déclenchée manuellement par admin
    """

    # Sécurité
    reviewed_by_admin = db.Column(Boolean, default=False)
    reviewed_at = db.Column(db.DateTime, nullable=True)
    admin_notes = db.Column(Text, nullable=True)

    # Métadonnées
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    updated_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relations
    company = db.relationship(
        "Company", backref=db.backref("autonomous_actions", lazy="dynamic")
    )
    booking = db.relationship(
        "Booking", backref=db.backref("autonomous_actions", lazy="dynamic")
    )
    driver = db.relationship(
        "Driver", backref=db.backref("autonomous_actions", lazy="dynamic")
    )

    @override
    def __repr__(self) -> str:
        return f"<AutonomousAction id={self.id} type={self.action_type} company={self.company_id} success={self.success}>"

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour API."""
        return {
            "id": self.id,
            "company_id": self.company_id,
            "booking_id": self.booking_id,
            "driver_id": self.driver_id,
            "action_type": self.action_type,
            "action_description": self.action_description,
            "action_data": self.action_data,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "confidence_score": self.confidence_score,
            "expected_improvement_minutes": self.expected_improvement_minutes,
            "trigger_source": self.trigger_source,
            "reviewed_by_admin": self.reviewed_by_admin,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "admin_notes": self.admin_notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def count_actions_last_hour(
        cls, company_id: int, action_type: str | None = None
    ) -> int:
        """Compte le nombre d'actions dans la dernière heure.

        Args:
            company_id: ID de l'entreprise
            action_type: Type d'action spécifique (optionnel)

        Returns:
            Nombre d'actions dans la dernière heure

        """
        from datetime import timedelta

        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

        query = cls.query.filter(
            cls.company_id == company_id,
            cls.created_at >= one_hour_ago,
            cls.success == True,  # noqa: E712
            # Exclure les actions de monitoring (tick) du comptage
            cls.action_type != "tick",
        )

        if action_type:
            query = query.filter(cls.action_type == action_type)

        return query.count()

    @classmethod
    def count_actions_today(
        cls, company_id: int, action_type: str | None = None
    ) -> int:
        """Compte le nombre d'actions aujourd'hui.

        Args:
            company_id: ID de l'entreprise
            action_type: Type d'action spécifique (optionnel)

        Returns:
            Nombre d'actions aujourd'hui

        """
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        query = cls.query.filter(
            cls.company_id == company_id,
            cls.created_at >= today_start,
            cls.success == True,  # noqa: E712
            # Exclure les actions de monitoring (tick) du comptage
            cls.action_type != "tick",
        )

        if action_type:
            query = query.filter(cls.action_type == action_type)

        return query.count()
