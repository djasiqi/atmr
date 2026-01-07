"""Modèles Swagger standardisés pour les erreurs API."""

from flask_restx import fields  # pyright: ignore[reportMissingImports]


def create_api_error_model(api):
    """Crée le modèle Swagger standardisé pour les erreurs API.

    Args:
        api: Instance Flask-RESTX API

    Returns:
        Modèle Swagger pour APIError
    """
    return api.model(
        "APIError",
        {
            "error": fields.String(
                required=True,
                description="Type d'erreur (ex: 'validation_error', 'not_found')",
                example="validation_error",
            ),
            "message": fields.String(
                required=True,
                description="Message d'erreur lisible par l'utilisateur",
                example="Le champ 'email' est requis",
            ),
            "trace_id": fields.String(
                required=False,
                description="ID de traçage pour le support technique",
                example="a1b2c3d4e5f6g7h8",
            ),
            "details": fields.Raw(
                required=False,
                description="Détails supplémentaires pour le debugging",
                example={"field": "email", "provided": None, "expected": "string"},
            ),
        },
    )


def create_validation_error_model(api):
    """Crée le modèle Swagger pour les erreurs de validation.

    Args:
        api: Instance Flask-RESTX API

    Returns:
        Modèle Swagger pour ValidationError
    """
    return api.model(
        "ValidationError",
        {
            "error": fields.String(
                required=True,
                description="Type d'erreur",
                example="validation_error",
            ),
            "message": fields.String(
                required=True,
                description="Message d'erreur",
                example="Erreur de validation",
            ),
            "trace_id": fields.String(
                required=False,
                description="ID de traçage",
            ),
            "fields": fields.Raw(
                required=False,
                description="Détails des erreurs par champ",
                example={"email": ["Ce champ est requis"], "password": ["Trop court"]},
            ),
        },
    )


def create_not_found_error_model(api):
    """Crée le modèle Swagger pour les erreurs 404.

    Args:
        api: Instance Flask-RESTX API

    Returns:
        Modèle Swagger pour NotFoundError
    """
    return api.model(
        "NotFoundError",
        {
            "error": fields.String(
                required=True,
                description="Type d'erreur",
                example="not_found",
            ),
            "message": fields.String(
                required=True,
                description="Message d'erreur",
                example="Ressource non trouvée",
            ),
            "trace_id": fields.String(
                required=False,
                description="ID de traçage",
            ),
            "resource": fields.String(
                required=False,
                description="Type de ressource non trouvée",
                example="Client",
            ),
            "resource_id": fields.String(
                required=False,
                description="ID de la ressource non trouvée",
                example="123",
            ),
        },
    )


def create_permission_error_model(api):
    """Crée le modèle Swagger pour les erreurs 403.

    Args:
        api: Instance Flask-RESTX API

    Returns:
        Modèle Swagger pour PermissionError
    """
    return api.model(
        "PermissionError",
        {
            "error": fields.String(
                required=True,
                description="Type d'erreur",
                example="permission_denied",
            ),
            "message": fields.String(
                required=True,
                description="Message d'erreur",
                example="Accès refusé",
            ),
            "trace_id": fields.String(
                required=False,
                description="ID de traçage",
            ),
            "required_role": fields.String(
                required=False,
                description="Rôle requis pour cette action",
                example="admin",
            ),
        },
    )
