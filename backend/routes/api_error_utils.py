"""✅ Utilitaire pour créer des réponses d'erreur API avec détails.

Fournit des helpers pour créer des messages d'erreur structurés avec :
- Un message principal clair pour l'utilisateur
- Des détails techniques pour le debugging
- Un code d'erreur standardisé
- Des suggestions de résolution quand approprié
"""

import traceback
from typing import Any, Dict, Tuple

from flask import current_app  # pyright: ignore[reportMissingImports]


def create_error_response(
    message: str,
    status_code: int = 500,
    *,
    error_code: str | None = None,
    details: Dict[str, Any] | None = None,
    suggestion: str | None = None,
    exception: Exception | None = None,
) -> Tuple[Dict[str, Any], int]:
    """Crée une réponse d'erreur structurée pour l'API.

    Args:
        message: Message d'erreur principal (compréhensible par l'utilisateur)
        status_code: Code HTTP de statut (défaut: 500)
        error_code: Code d'erreur standardisé (ex: "validation_error", "not_found")
        details: Détails supplémentaires pour le debugging (dict)
        suggestion: Suggestion de résolution pour l'utilisateur
        exception: Exception originale (pour extraire des détails en dev)

    Returns:
        Tuple (response_json, status_code) pour Flask

    Examples:
        # Erreur simple
        return create_error_response("Paramètre manquant", 400)

        # Erreur avec détails
        return create_error_response(
            "Format de date invalide",
            400,
            error_code="invalid_date_format",
            details={"provided": "2025-13-45", "expected": "YYYY-MM-DD"},
            suggestion="Utilisez le format YYYY-MM-DD (ex: 2025-12-25)"
        )

        # Erreur avec exception
        return create_error_response(
            "Échec de l'opération",
            500,
            error_code="internal_error",
            exception=e
        )
    """
    response: Dict[str, Any] = {
        "error": message,
    }

    # Ajouter le code d'erreur si fourni
    if error_code:
        response["error_code"] = error_code

    # Ajouter les détails si fournis
    if details:
        response["details"] = details

    # Ajouter la suggestion si fournie
    if suggestion:
        response["suggestion"] = suggestion

    # En mode développement, ajouter des détails de l'exception
    if exception and current_app and current_app.config.get("DEBUG", False):
        response["debug"] = {
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback.format_exc().split("\n"),
        }

    return response, status_code


def create_validation_error(
    message: str,
    field: str | None = None,
    provided_value: Any | None = None,
    expected_format: str | None = None,
) -> Tuple[Dict[str, Any], int]:
    """Crée une réponse d'erreur de validation.

    Args:
        message: Message d'erreur principal
        field: Nom du champ en erreur
        provided_value: Valeur fournie (si applicable)
        expected_format: Format attendu (si applicable)

    Returns:
        Tuple (response_json, status_code) pour Flask
    """
    details: Dict[str, Any] = {}
    if field:
        details["field"] = field
    if provided_value is not None:
        details["provided_value"] = provided_value
    if expected_format:
        details["expected_format"] = expected_format

    suggestion = None
    if expected_format:
        suggestion = f"Utilisez le format: {expected_format}"

    return create_error_response(
        message,
        status_code=400,
        error_code="validation_error",
        details=details if details else None,
        suggestion=suggestion,
    )


def create_not_found_error(
    resource_type: str,
    resource_id: Any | None = None,
) -> Tuple[Dict[str, Any], int]:
    """Crée une réponse d'erreur 404 (ressource non trouvée).

    Args:
        resource_type: Type de ressource (ex: "Entreprise", "Réservation")
        resource_id: ID de la ressource recherchée (optionnel)

    Returns:
        Tuple (response_json, status_code) pour Flask
    """
    message = f"{resource_type} introuvable"
    if resource_id is not None:
        message = f"{resource_type} avec l'ID '{resource_id}' introuvable"

    details: Dict[str, Any] = {"resource_type": resource_type}
    if resource_id is not None:
        details["resource_id"] = resource_id

    return create_error_response(
        message,
        status_code=404,
        error_code="not_found",
        details=details,
        suggestion=f"Vérifiez que le {resource_type.lower()} existe et que vous avez les permissions nécessaires.",
    )


def create_internal_error(
    message: str = "Une erreur interne s'est produite",
    operation: str | None = None,
    exception: Exception | None = None,
) -> Tuple[Dict[str, Any], int]:
    """Crée une réponse d'erreur 500 (erreur interne).

    Args:
        message: Message d'erreur principal
        operation: Nom de l'opération qui a échoué (ex: "démarrage", "arrêt")
        exception: Exception originale (pour détails en dev)

    Returns:
        Tuple (response_json, status_code) pour Flask
    """
    error_message = message
    if operation:
        error_message = f"Échec de {operation}: {message}"

    details: Dict[str, Any] = {}
    if operation:
        details["operation"] = operation

    return create_error_response(
        error_message,
        status_code=500,
        error_code="internal_error",
        details=details if details else None,
        suggestion="Veuillez réessayer plus tard. Si le problème persiste, contactez le support.",
        exception=exception,
    )


def create_permission_error(
    message: str = "Permission refusée",
    required_permission: str | None = None,
) -> Tuple[Dict[str, Any], int]:
    """Crée une réponse d'erreur 403 (permission refusée).

    Args:
        message: Message d'erreur principal
        required_permission: Permission requise (optionnel)

    Returns:
        Tuple (response_json, status_code) pour Flask
    """
    details: Dict[str, Any] = {}
    if required_permission:
        details["required_permission"] = required_permission

    return create_error_response(
        message,
        status_code=403,
        error_code="permission_denied",
        details=details if details else None,
        suggestion="Vérifiez que vous avez les permissions nécessaires pour effectuer cette action.",
    )


def create_rate_limit_error(
    retry_after: int | None = None,
) -> Tuple[Dict[str, Any], int]:
    """Crée une réponse d'erreur 429 (rate limit).

    Args:
        retry_after: Nombre de secondes avant de pouvoir réessayer

    Returns:
        Tuple (response_json, status_code) pour Flask
    """
    message = "Trop de requêtes. Veuillez patienter avant de réessayer."
    details: Dict[str, Any] = {}
    if retry_after:
        details["retry_after_seconds"] = retry_after
        message = f"Trop de requêtes. Réessayez dans {retry_after} secondes."

    return create_error_response(
        message,
        status_code=429,
        error_code="rate_limit_exceeded",
        details=details if details else None,
        suggestion="Attendez quelques instants avant de réessayer.",
    )


def create_conflict_error(
    message: str,
    resource_type: str | None = None,
    resource_id: Any | None = None,
) -> Tuple[Dict[str, Any], int]:
    """Crée une réponse d'erreur 409 (conflict).

    Args:
        message: Message d'erreur principal
        resource_type: Type de ressource en conflit (optionnel)
        resource_id: ID de la ressource en conflit (optionnel)

    Returns:
        Tuple (response_json, status_code) pour Flask
    """
    details: Dict[str, Any] = {}
    if resource_type:
        details["resource_type"] = resource_type
    if resource_id is not None:
        details["resource_id"] = resource_id

    return create_error_response(
        message,
        status_code=409,
        error_code="conflict",
        details=details if details else None,
        suggestion="La ressource existe déjà ou est en conflit avec l'état actuel. Vérifiez l'état avant de réessayer.",
    )
