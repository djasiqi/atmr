# backend/tests/helpers/rollback_verification.py
"""Helpers pour vérifier que les rollbacks restaurent bien les valeurs.

Ce module fournit des utilitaires pour vérifier systématiquement que les rollbacks
SQLAlchemy restaurent correctement les valeurs en DB.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Type

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def verify_rollback_restores_values(
    db_session: Session,
    model_class: Type[Any],
    object_id: int,
    original_values: Dict[str, Any],
    *,
    reload_strategy: str = "query",
) -> bool:
    """Vérifie qu'un rollback a bien restauré les valeurs originales en DB.

    Cette fonction vérifie que les valeurs d'un objet en DB correspondent
    aux valeurs originales après un rollback.

    Args:
        db_session: Session SQLAlchemy
        model_class: Classe du modèle SQLAlchemy (ex: Booking, Assignment)
        object_id: ID de l'objet à vérifier
        original_values: Dictionnaire des valeurs originales {field: value}
        reload_strategy: Stratégie de rechargement ("query" ou "get")

    Returns:
        True si toutes les valeurs correspondent, False sinon

    Raises:
        AssertionError: Si l'objet n'existe pas ou si les valeurs ne correspondent pas

    Example:
        ```python
        # Avant modification
        booking = BookingFactory(company=company, driver_id=None)
        db.session.commit()
        original_values = {"driver_id": None, "status": BookingStatus.ACCEPTED}

        # Modifier
        booking.driver_id = driver.id
        booking.status = BookingStatus.ASSIGNED
        db.session.flush()

        # Rollback
        db.session.rollback()
        db.session.expire_all()

        # Vérifier
        verify_rollback_restores_values(
            db.session,
            Booking,
            booking.id,
            original_values,
        )
        ```
    """
    # Expirer tous les objets pour forcer un rechargement depuis la DB
    db_session.expire_all()

    # Recharger l'objet depuis la DB
    if reload_strategy == "query":
        reloaded = db_session.query(model_class).filter_by(id=object_id).first()
    elif reload_strategy == "get":
        reloaded = db_session.query(model_class).get(object_id)
    else:
        raise ValueError(
            f"Invalid reload_strategy: {reload_strategy}. Use 'query' or 'get'"
        )

    if reloaded is None:
        error_msg = (
            f"{model_class.__name__} with id={object_id} not found in DB after rollback. "
            f"Object may have been deleted or never committed."
        )
        raise AssertionError(error_msg)

    # Vérifier chaque valeur originale
    mismatches: List[str] = []
    for field, expected_value in original_values.items():
        actual_value = getattr(reloaded, field, None)
        if actual_value != expected_value:
            mismatches.append(
                f"{field}: expected={expected_value!r}, actual={actual_value!r}"
            )

    if mismatches:
        error_msg = (
            f"Rollback did not restore original values for {model_class.__name__} id={object_id}:\n"
            f"{chr(10).join(f'  - {mismatch}' for mismatch in mismatches)}"
        )
        logger.error(error_msg)
        raise AssertionError(error_msg)

    logger.debug(
        "✅ Rollback verified: %s id=%s values restored correctly",
        model_class.__name__,
        object_id,
    )
    return True


def capture_original_values(
    obj: Any, fields: List[str] | None = None
) -> Dict[str, Any]:
    """Capture les valeurs originales d'un objet avant modification.

    Args:
        obj: Objet SQLAlchemy
        fields: Liste des champs à capturer (None = tous les attributs non privés)

    Returns:
        Dictionnaire des valeurs originales {field: value}

    Example:
        ```python
        booking = BookingFactory(company=company, driver_id=None)
        db.session.commit()

        original_values = capture_original_values(booking, ["driver_id", "status"])
        # Modifier...
        # Rollback...
        # Vérifier avec verify_rollback_restores_values()
        ```
    """
    if fields is None:
        # Capturer tous les attributs non privés
        fields = [
            attr
            for attr in dir(obj)
            if not attr.startswith("_") and not callable(getattr(obj, attr, None))
        ]

    original_values: Dict[str, Any] = {}
    for field in fields:
        if hasattr(obj, field):
            original_values[field] = getattr(obj, field)

    return original_values


def verify_multiple_rollbacks(
    db_session: Session,
    verifications: List[Dict[str, Any]],
) -> bool:
    """Vérifie plusieurs rollbacks en une seule opération.

    Args:
        db_session: Session SQLAlchemy
        verifications: Liste de dictionnaires avec les clés :
            - model_class: Classe du modèle
            - object_id: ID de l'objet
            - original_values: Dictionnaire des valeurs originales

    Returns:
        True si tous les rollbacks sont corrects, False sinon

    Raises:
        AssertionError: Si un rollback n'a pas restauré les valeurs

    Example:
        ```python
        verifications = [
            {
                "model_class": Booking,
                "object_id": booking1.id,
                "original_values": {"driver_id": None},
            },
            {
                "model_class": Booking,
                "object_id": booking2.id,
                "original_values": {"driver_id": None},
            },
        ]
        verify_multiple_rollbacks(db.session, verifications)
        ```
    """
    all_correct = True
    errors: List[str] = []

    for verification in verifications:
        try:
            verify_rollback_restores_values(
                db_session,
                verification["model_class"],
                verification["object_id"],
                verification["original_values"],
                reload_strategy=verification.get("reload_strategy", "query"),
            )
        except AssertionError as e:
            all_correct = False
            errors.append(str(e))

    if not all_correct:
        error_msg = "Multiple rollback verifications failed:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
        raise AssertionError(error_msg)

    return True
