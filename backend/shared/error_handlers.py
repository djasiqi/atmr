"""Handler centralisé pour la gestion des erreurs API.

Les exceptions techniques ne sont jamais renvoyées au client (ni str, ni type,
ni traceback). Les messages HTTP sont des constantes contrôlées.
"""

import logging
from typing import Any

from marshmallow import ValidationError
from sqlalchemy.exc import IntegrityError
from werkzeug.exceptions import HTTPException

from routes.api_error_utils import (
    create_billing_validation_error,
    create_conflict_error,
    create_error_response,
    create_internal_error,
    create_not_found_error,
    create_permission_error,
    create_service_unavailable_error,
    create_validation_error,
)
from routes.db_error_utils import format_integrity_error
from shared.logging_utils import exception_type_for_log

logger = logging.getLogger(__name__)

_HTTP_STATUS_MESSAGES: dict[int, tuple[str, str]] = {
    400: ("bad_request", "Requête invalide"),
    401: ("unauthorized", "Authentification requise"),
    403: ("forbidden", "Permission refusée"),
    404: ("not_found", "Ressource introuvable"),
    405: ("method_not_allowed", "Méthode non autorisée"),
    409: ("conflict", "Conflit"),
    422: ("unprocessable_entity", "Données invalides"),
    429: ("too_many_requests", "Trop de requêtes"),
}


def _log_error_type(
    log: logging.Logger, event: str, exception: Exception, level: int = logging.WARNING
) -> None:
    log.log(level, "%s error_type=%s", event, exception_type_for_log(exception))


class APIErrorHandler:
    """Handler centralisé pour la gestion des erreurs API."""

    @staticmethod
    def handle_exception(
        exception: Exception,
        logger_instance: logging.Logger | None = None,
        default_message: str = "Une erreur interne s'est produite",
    ) -> tuple[dict[str, Any], int]:
        """Convertit une exception en réponse HTTP sans fuite technique.

        HTTPException est relancée pour le handler Flask (status + message stables).
        """
        log = logger_instance or logger

        if isinstance(exception, HTTPException):
            raise exception

        if isinstance(exception, ValidationError):
            _log_error_type(log, "api_validation_error", exception)
            return create_validation_error("Données invalides")

        if isinstance(exception, IntegrityError):
            _log_error_type(log, "api_integrity_error", exception)
            return format_integrity_error(exception)

        if isinstance(exception, ValueError):
            _log_error_type(log, "api_value_error", exception)
            return create_validation_error("Données invalides")

        if isinstance(exception, KeyError):
            _log_error_type(log, "api_key_error", exception)
            return create_validation_error("Champ manquant")

        _log_error_type(log, "unhandled_api_error", exception, level=logging.ERROR)
        return create_internal_error(
            default_message,
            exception=exception,
            operation="opération API",
        )

    @staticmethod
    def handle_http_status(status_code: int) -> tuple[dict[str, Any], int]:
        """Message HTTP stable (sans description Werkzeug / path)."""
        error_code, message = _HTTP_STATUS_MESSAGES.get(
            status_code, ("http_error", "Erreur HTTP")
        )
        return {"error": error_code, "message": message}, status_code

    @staticmethod
    def handle_not_found(
        resource_type: str,
        resource_id: Any | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        log = logger_instance or logger
        log.warning(
            "%s non trouvé%s",
            resource_type,
            f" (ID: {resource_id})" if resource_id is not None else "",
        )
        return create_not_found_error(resource_type, resource_id)

    @staticmethod
    def handle_not_found_error(
        message: str,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        log = logger_instance or logger
        log.warning("Not found")
        return create_error_response(message, 404, error_code="not_found")

    @staticmethod
    def handle_validation_error(
        message: str,
        field: str | None = None,
        provided_value: Any | None = None,
        expected_format: str | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        log = logger_instance or logger
        log.warning("Erreur de validation field=%s", field)
        return create_validation_error(
            message,
            field=field,
            provided_value=provided_value,
            expected_format=expected_format,
        )

    @staticmethod
    def handle_billing_validation_error(
        message: str,
        field: str | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        log = logger_instance or logger
        log.warning("Facturation invalide field=%s", field)
        return create_billing_validation_error(message, field=field)

    @staticmethod
    def handle_permission_error(
        message: str = "Permission refusée",
        required_permission: str | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        log = logger_instance or logger
        log.warning("Permission refusée")
        return create_permission_error(message, required_permission=required_permission)

    @staticmethod
    def handle_conflict_error(
        message: str,
        resource_type: str | None = None,
        resource_id: Any | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        log = logger_instance or logger
        log.warning("Conflit détecté resource_type=%s", resource_type)
        return create_conflict_error(
            message, resource_type=resource_type, resource_id=resource_id
        )

    @staticmethod
    def handle_service_unavailable_error(
        message: str,
        error_code: str = "service_unavailable",
        details: dict[str, Any] | None = None,
        logger_instance: logging.Logger | None = None,
    ) -> tuple[dict[str, Any], int]:
        log = logger_instance or logger
        log.error("Service indisponible error_code=%s", error_code)
        return create_service_unavailable_error(
            message, error_code=error_code, details=details
        )
