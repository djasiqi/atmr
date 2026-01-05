"""Handler centralisé pour la gestion des erreurs API.

Ce module fournit un handler unifié pour gérer les exceptions dans les routes,
améliorant la cohérence des réponses d'erreur et simplifiant la maintenance.
"""

import logging
from typing import Any

from marshmallow import ValidationError  # pyright: ignore[reportMissingImports]
from sqlalchemy.exc import IntegrityError

from routes.api_error_utils import (
    create_conflict_error,
    create_internal_error,
    create_not_found_error,
    create_permission_error,
    create_validation_error,
)
from routes.db_error_utils import format_integrity_error

logger = logging.getLogger(__name__)


class APIErrorHandler:
    """Handler centralisé pour la gestion des erreurs API."""

    @staticmethod
    def handle_exception(
        exception: Exception,
        logger_instance: logging.Logger | None = None,
        default_message: str = "Une erreur interne s'est produite",
    ) -> tuple[dict[str, Any], int]:
        """Gère une exception et retourne une réponse d'erreur standardisée.

        Args:
            exception: Exception à gérer
            logger_instance: Logger à utiliser (défaut: logger du module)
            default_message: Message par défaut si l'exception n'est pas reconnue

        Returns:
            Tuple (response_json, status_code) pour Flask

        Examples:
            # Dans une route
            try:
                # ...
            except Exception as e:
                return APIErrorHandler.handle_exception(e, app_logger)
        """
        log = logger_instance or logger
        result: tuple[dict[str, Any], int] | None = None

        # ValidationError (Marshmallow)
        if isinstance(exception, ValidationError):
            log.warning("Erreur de validation: %s", exception.messages)
            # Extraire le premier message d'erreur
            error_messages = exception.messages
            if isinstance(error_messages, dict):
                # Prendre le premier champ en erreur
                first_field = next(iter(error_messages.keys()))
                first_error = error_messages[first_field]
                if isinstance(first_error, list) and first_error:
                    message = f"{first_field}: {first_error[0]}"
                else:
                    message = str(first_error)
            else:
                message = str(error_messages)
            result = create_validation_error(message)

        # IntegrityError (SQLAlchemy)
        elif isinstance(exception, IntegrityError):
            log.warning("Erreur d'intégrité DB: %s", exception)
            result = format_integrity_error(exception)

        # ValueError (généralement erreur de validation)
        elif isinstance(exception, ValueError):
            log.warning("Erreur de valeur: %s", exception)
            result = create_validation_error(str(exception))

        # KeyError (clé manquante dans dict)
        elif isinstance(exception, KeyError):
            log.warning("Clé manquante: %s", exception)
            result = create_validation_error(
                f"Champ manquant: {exception}", field=str(exception)
            )

        # AttributeError (attribut manquant)
        elif isinstance(exception, AttributeError):
            log.warning("Attribut manquant: %s", exception)
            result = create_validation_error(f"Attribut manquant: {exception}")

        # FileNotFoundError
        elif isinstance(exception, FileNotFoundError):
            log.warning("Fichier non trouvé: %s", exception)
            result = create_not_found_error("Fichier", str(exception))

        # PermissionError
        elif isinstance(exception, PermissionError):
            log.warning("Permission refusée: %s", exception)
            result = create_permission_error(str(exception))

        # Erreur générique - logger l'exception complète
        if result is None:
            log.exception("Erreur serveur non gérée: %s", exception)
            result = create_internal_error(
                default_message,
                exception=exception,
                operation="opération API",
            )

        return result

    @staticmethod
    def handle_not_found(
        resource_type: str,
        resource_id: Any | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Gère une erreur 404 (ressource non trouvée).

        Args:
            resource_type: Type de ressource (ex: "Entreprise", "Réservation")
            resource_id: ID de la ressource recherchée (optionnel)
            logger_instance: Logger à utiliser (défaut: logger du module)

        Returns:
            Tuple (response_json, status_code) pour Flask
        """
        log = logger_instance or logger
        log.warning(
            "%s non trouvé%s",
            resource_type,
            f" (ID: {resource_id})" if resource_id is not None else "",
        )
        return create_not_found_error(resource_type, resource_id)

    @staticmethod
    def handle_validation_error(
        message: str,
        field: str | None = None,
        provided_value: Any | None = None,
        expected_format: str | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Gère une erreur de validation.

        Args:
            message: Message d'erreur principal
            field: Nom du champ en erreur (optionnel)
            provided_value: Valeur fournie (optionnel)
            expected_format: Format attendu (optionnel)
            logger_instance: Logger à utiliser (défaut: logger du module)

        Returns:
            Tuple (response_json, status_code) pour Flask
        """
        log = logger_instance or logger
        log.warning("Erreur de validation: %s (champ: %s)", message, field)
        return create_validation_error(
            message,
            field=field,
            provided_value=provided_value,
            expected_format=expected_format,
        )

    @staticmethod
    def handle_permission_error(
        message: str = "Permission refusée",
        required_permission: str | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Gère une erreur de permission (403).

        Args:
            message: Message d'erreur principal
            required_permission: Permission requise (optionnel)
            logger_instance: Logger à utiliser (défaut: logger du module)

        Returns:
            Tuple (response_json, status_code) pour Flask
        """
        log = logger_instance or logger
        log.warning("Permission refusée: %s", message)
        return create_permission_error(message, required_permission=required_permission)

    @staticmethod
    def handle_conflict_error(
        message: str,
        resource_type: str | None = None,
        resource_id: Any | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        """Gère une erreur de conflit (409).

        Args:
            message: Message d'erreur principal
            resource_type: Type de ressource en conflit (optionnel)
            resource_id: ID de la ressource en conflit (optionnel)
            logger_instance: Logger à utiliser (défaut: logger du module)

        Returns:
            Tuple (response_json, status_code) pour Flask
        """
        log = logger_instance or logger
        log.warning(
            "Conflit détecté: %s%s",
            message,
            f" (ressource: {resource_type}, ID: {resource_id})"
            if resource_type
            else "",
        )
        return create_conflict_error(
            message, resource_type=resource_type, resource_id=resource_id
        )
