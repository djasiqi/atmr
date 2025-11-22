"""Modèle SecretRotation pour stocker l'historique des rotations de secrets via Vault.

Permet de suivre toutes les rotations de secrets (succès, échecs, skips) pour
le monitoring et l'audit de sécurité.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from ext import db


class SecretRotation(db.Model):
    """Stocke l'historique des rotations de secrets via Vault.

    Enregistre chaque tentative de rotation (succès, échec, skip) pour :
    - Monitoring et observabilité
    - Audit de sécurité
    - Statistiques et métriques
    - Détection d'anomalies
    """

    __tablename__ = "secret_rotation"

    id = Column(Integer, primary_key=True)

    # Type de secret roté
    secret_type = Column(String(50), nullable=False, index=True)
    """
    Types possibles:
    - 'jwt': Clé secrète JWT
    - 'encryption': Clé d'encryption maître
    - 'flask_secret_key': SECRET_KEY Flask
    """

    # Statut de la rotation
    status = Column(String(20), nullable=False, index=True)
    """
    Statuts possibles:
    - 'success': Rotation réussie
    - 'error': Erreur lors de la rotation
    - 'skipped': Rotation ignorée (Vault non disponible, etc.)
    """

    # Timestamp de la rotation
    rotated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )

    # Environnement
    environment = Column(String(20), nullable=False, index=True)
    """
    Environnements possibles:
    - 'dev': Développement
    - 'prod': Production
    - 'testing': Tests
    """

    # Métadonnées additionnelles (JSONB pour flexibilité)
    # Note: Utiliser db.Column pour éviter conflit avec Model.metadata
    rotation_metadata = db.Column(JSONB, nullable=True)
    """
    Métadonnées optionnelles selon le type de secret:
    - next_rotation_days: Nombre de jours avant prochaine rotation
    - legacy_keys_count: Nombre de clés legacy (pour encryption)
    - old_secret_present: Si une ancienne clé existait
    - task_id: ID de la tâche Celery
    """

    # Message d'erreur si status='error'
    error_message = Column(Text, nullable=True)

    # ID de la tâche Celery (optionnel)
    task_id = Column(String(255), nullable=True, index=True)

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire pour export API."""
        # rotated_at est nullable=False mais peut être None avant commit
        rotated_at_value: datetime | None = self.rotated_at
        return {
            "id": self.id,
            "secret_type": self.secret_type,
            "status": self.status,
            "rotated_at": rotated_at_value.isoformat()
            if rotated_at_value is not None
            else None,
            "environment": self.environment,
            "metadata": self.rotation_metadata,
            "error_message": self.error_message,
            "task_id": self.task_id,
        }

    def __repr__(self) -> str:  # pyright: ignore[reportImplicitOverride]
        return (
            f"<SecretRotation id={self.id} type={self.secret_type} "
            f"status={self.status} env={self.environment} at={self.rotated_at}>"
        )
