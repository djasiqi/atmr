"""✅ Utilitaire centralisé pour validation Marshmallow avec erreurs 400 détaillées.

Fournit des helpers pour valider les entrées et retourner des erreurs structurées.
"""

from typing import Any, Dict, cast

from marshmallow import Schema, ValidationError
from marshmallow.validate import Length


def validate_request(schema: Schema, data: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
    """Valide les données de requête avec un schema Marshmallow.

    Args:
        schema: Schema Marshmallow à utiliser pour la validation
        data: Données à valider (dict)
        strict: Si True, rejette les champs inconnus (défaut: True)

    Returns:
        Dict validé et nettoyé

    Raises:
        ValidationError: Si la validation échoue (avec détails par champ)
    """
    try:
        # Validation stricte: rejette les champs inconnus par défaut
        validated = schema.load(data, unknown="EXCLUDE" if strict else "INCLUDE")
        # Cast pour type checker (schema.load retourne Any)
        return cast(Dict[str, Any], validated)
    except ValidationError as err:
        # Format personnalisé pour erreurs 400 détaillées
        formatted_errors = _format_validation_errors(cast(Dict[str, Any], err.messages))
        raise ValidationError(formatted_errors) from err


def _format_validation_errors(errors: Dict[str, Any]) -> Dict[str, Any]:
    """Formate les erreurs de validation en structure standardisée.

    Structure retournée:
    {
        "message": "Erreur de validation",
        "errors": {
            "field_name": ["message d'erreur 1", "message d'erreur 2"],
            ...
        }
    }

    Args:
        errors: Messages d'erreur bruts de Marshmallow

    Returns:
        Dict formaté avec structure standardisée
    """
    formatted: Dict[str, Any] = {"message": "Erreur de validation des données", "errors": {}}

    for field, messages in errors.items():
        # ⚡ Ignorer les clés spéciales de Marshmallow (_schema, _nested, etc.)
        # qui ne sont pas des champs de formulaire
        if field.startswith("_"):
            # Si c'est une erreur au niveau du schéma, l'ajouter au message général
            if field == "_schema" and isinstance(messages, list):
                formatted["message"] = messages[0] if messages else "Erreur de validation des données"
            continue

        # Si c'est une liste de messages, prendre directement
        if isinstance(messages, list):
            formatted["errors"][field] = messages
        # Si c'est un dict (champs nested ou erreurs de liste), formater récursivement
        elif isinstance(messages, dict):
            # ⚡ Détecter si c'est une erreur de validation de liste (clés = indices entiers)
            # Exemple: {'client_ids': {1: ['Must be greater than or equal to 1.']}}
            all_keys_are_int_indices = all(
                (isinstance(k, int) or (isinstance(k, str) and k.isdigit())) for k in messages
            )

            if all_keys_are_int_indices:
                # ⚡ Cas: erreur de validation de liste (regrouper toutes les erreurs sous le nom du champ)
                all_list_errors: list[str] = []
                for _, index_msgs in messages.items():
                    if isinstance(index_msgs, list):
                        all_list_errors.extend(index_msgs)
                    else:
                        all_list_errors.append(str(index_msgs))
                formatted["errors"][field] = all_list_errors
            # ⚡ Éviter de créer une structure errors.errors.errors...
            # Si le dict contient déjà "errors" ou "message", extraire directement les champs
            elif "errors" in messages and isinstance(messages["errors"], dict):
                # Cas: erreur nested avec structure {errors: {...}}
                for nested_field, nested_msgs in messages["errors"].items():
                    if isinstance(nested_msgs, list):
                        formatted["errors"][f"{field}.{nested_field}"] = nested_msgs
                    else:
                        formatted["errors"][f"{field}.{nested_field}"] = [str(nested_msgs)]
            else:
                # Formatage normal récursif pour champs nested
                nested_formatted = _format_validation_errors(cast(Dict[str, Any], messages))
                # Fusionner les erreurs nested directement dans formatted["errors"]
                if "errors" in nested_formatted:
                    for nested_field, nested_msgs in nested_formatted["errors"].items():
                        formatted["errors"][f"{field}.{nested_field}"] = nested_msgs
        # Sinon, convertir en liste
        else:
            formatted["errors"][field] = [str(messages)]

    return formatted


def handle_validation_error(error: ValidationError):
    """Gère une ValidationError et retourne une réponse Flask 400.

    Usage:
        try:
            data = validate_request(schema, request.get_json())
        except ValidationError as e:
            return handle_validation_error(e)

    Args:
        error: Exception ValidationError de Marshmallow

    Returns:
        Tuple (response_json, status_code) pour Flask
    """
    formatted = _format_validation_errors(cast(Dict[str, Any], error.messages))
    # Retourner un dict brut pour laisser Flask-RESTX sérialiser correctement
    return formatted, 400


# Validators personnalisés réutilisables
EMAIL_VALIDATOR = Length(min=5, max=254, error="Email doit faire entre 5 et 254 caractères")
USERNAME_VALIDATOR = Length(min=3, max=50, error="Username doit faire entre 3 et 50 caractères")
PASSWORD_VALIDATOR = Length(min=8, error="Mot de passe doit faire au moins 8 caractères")
PHONE_VALIDATOR = Length(min=10, max=20, error="Téléphone doit faire entre 10 et 20 caractères")

# Formats de validation courants
ISO8601_DATE_REGEX = r"^\d{4}-\d{2}-\d{2}$"
ISO8601_DATETIME_REGEX = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?(Z|[+-]\d{2}:\d{2})?$"
