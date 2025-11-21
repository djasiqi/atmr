"""Service de monitoring pour les rotations de secrets via Vault.

Fournit des fonctions pour enregistrer et consulter l'historique des rotations
de secrets (JWT, encryption, Flask SECRET_KEY).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

from ext import db
from models import SecretRotation

logger = logging.getLogger(__name__)

# Types de secrets supportés
SECRET_TYPES = {"jwt", "encryption", "flask_secret_key"}

# Statuts possibles
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_SKIPPED = "skipped"


def record_rotation(
    secret_type: str,
    status: str,
    environment: str,
    metadata: dict[str, Any] | None = None,
    error_message: str | None = None,
    task_id: str | None = None,
) -> SecretRotation:
    """Enregistre une rotation de secret dans la base de données.

    Args:
        secret_type: Type de secret ('jwt', 'encryption', 'flask_secret_key')
        status: Statut de la rotation ('success', 'error', 'skipped')
        environment: Environnement ('dev', 'prod', 'testing')
        metadata: Métadonnées additionnelles (JSON)
        error_message: Message d'erreur si status='error'
        task_id: ID de la tâche Celery (optionnel)

    Returns:
        Instance SecretRotation créée

    Raises:
        ValueError: Si secret_type ou status invalide
    """
    if secret_type not in SECRET_TYPES:
        msg = f"Invalid secret_type: {secret_type}. Must be one of {SECRET_TYPES}"
        raise ValueError(msg)

    if status not in {STATUS_SUCCESS, STATUS_ERROR, STATUS_SKIPPED}:
        msg = f"Invalid status: {status}. Must be one of {STATUS_SUCCESS, STATUS_ERROR, STATUS_SKIPPED}"
        raise ValueError(msg)

    try:
        # Calme le type checker sur kwargs SQLAlchemy (métaclasses)
        SecretRotationType = cast("Any", SecretRotation)
        rotation = SecretRotationType(
            secret_type=secret_type,
            status=status,
            rotated_at=datetime.now(UTC),
            environment=environment,
            rotation_metadata=metadata,
            error_message=error_message,
            task_id=task_id,
        )

        db.session.add(rotation)
        db.session.commit()

        logger.info(
            "[SecretRotationMonitor] ✅ Rotation enregistrée: type=%s, status=%s, env=%s",
            secret_type,
            status,
            environment,
        )

        return rotation

    except Exception as e:
        db.session.rollback()
        logger.exception("[SecretRotationMonitor] ❌ Erreur enregistrement rotation: %s", e)
        raise


def get_rotation_history(
    secret_type: str | None = None,
    status: str | None = None,
    environment: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[SecretRotation], int]:
    """Récupère l'historique des rotations avec filtres optionnels.

    Args:
        secret_type: Filtrer par type de secret
        status: Filtrer par statut
        environment: Filtrer par environnement
        limit: Nombre maximum de résultats
        offset: Offset pour pagination

    Returns:
        Tuple (liste des rotations, total count)
    """
    try:
        query = SecretRotation.query

        if secret_type:
            query = query.filter(SecretRotation.secret_type == secret_type)
        if status:
            query = query.filter(SecretRotation.status == status)
        if environment:
            query = query.filter(SecretRotation.environment == environment)

        # Compter le total avant pagination
        total = query.count()

        # Appliquer pagination et tri
        rotations = query.order_by(SecretRotation.rotated_at.desc()).limit(limit).offset(offset).all()

        return rotations, total

    except Exception as e:
        logger.exception("[SecretRotationMonitor] ❌ Erreur récupération historique: %s", e)
        raise


def get_rotation_stats() -> dict[str, Any]:
    """Calcule les statistiques globales des rotations.

    Returns:
        Dictionnaire avec statistiques par type et globales
    """
    try:
        # Statistiques globales
        total_rotations = SecretRotation.query.count()
        success_count = SecretRotation.query.filter(SecretRotation.status == STATUS_SUCCESS).count()
        error_count = SecretRotation.query.filter(SecretRotation.status == STATUS_ERROR).count()
        skipped_count = SecretRotation.query.filter(SecretRotation.status == STATUS_SKIPPED).count()

        # Statistiques par type
        by_type: dict[str, dict[str, int]] = {}
        for secret_type in SECRET_TYPES:
            type_query = SecretRotation.query.filter(SecretRotation.secret_type == secret_type)
            by_type[secret_type] = {
                "total": type_query.count(),
                "success": type_query.filter(SecretRotation.status == STATUS_SUCCESS).count(),
                "error": type_query.filter(SecretRotation.status == STATUS_ERROR).count(),
                "skipped": type_query.filter(SecretRotation.status == STATUS_SKIPPED).count(),
            }

        # Dernières rotations par type
        last_rotations: dict[str, str | None] = {}
        for secret_type in SECRET_TYPES:
            last = (
                SecretRotation.query.filter(SecretRotation.secret_type == secret_type)
                .order_by(SecretRotation.rotated_at.desc())
                .first()
            )
            last_rotations[secret_type] = last.rotated_at.isoformat() if last else None

        return {
            "total_rotations": total_rotations,
            "success_count": success_count,
            "error_count": error_count,
            "skipped_count": skipped_count,
            "by_type": by_type,
            "last_rotations": last_rotations,
        }

    except Exception as e:
        logger.exception("[SecretRotationMonitor] ❌ Erreur calcul statistiques: %s", e)
        raise


def get_last_rotation(secret_type: str, environment: str | None = None) -> SecretRotation | None:
    """Récupère la dernière rotation pour un type de secret donné.

    Args:
        secret_type: Type de secret
        environment: Filtrer par environnement (optionnel)

    Returns:
        Dernière rotation ou None si aucune trouvée
    """
    try:
        query = SecretRotation.query.filter(SecretRotation.secret_type == secret_type)

        if environment:
            query = query.filter(SecretRotation.environment == environment)

        return query.order_by(SecretRotation.rotated_at.desc()).first()

    except Exception as e:
        logger.exception("[SecretRotationMonitor] ❌ Erreur récupération dernière rotation: %s", e)
        raise


def get_days_since_last_rotation(secret_type: str, environment: str | None = None) -> int | None:
    """Calcule le nombre de jours depuis la dernière rotation réussie.

    Args:
        secret_type: Type de secret
        environment: Filtrer par environnement (optionnel)

    Returns:
        Nombre de jours ou None si aucune rotation réussie trouvée
    """
    try:
        query = SecretRotation.query.filter(
            SecretRotation.secret_type == secret_type, SecretRotation.status == STATUS_SUCCESS
        )

        if environment:
            query = query.filter(SecretRotation.environment == environment)

        last = query.order_by(SecretRotation.rotated_at.desc()).first()

        if not last or not last.rotated_at:
            return None

        delta = datetime.now(UTC) - last.rotated_at
        return delta.days

    except Exception as e:
        logger.exception("[SecretRotationMonitor] ❌ Erreur calcul jours depuis dernière rotation: %s", e)
        return None
