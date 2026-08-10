"""✅ Utilitaire centralisé pour validation Marshmallow avec erreurs 400 détaillées.

Fournit des helpers pour valider les entrées et retourner des erreurs structurées.
"""

import json
from typing import Any, Dict, cast

from flask import request
from marshmallow import (
    Schema,
    ValidationError,
)
from marshmallow.validate import Length


def parse_request_json() -> Dict[str, Any]:
    """Parse le corps JSON de la requête sans lever BadRequest Werkzeug.

    Flask-RESTX (validate=True) et ``request.get_json()`` sans ``silent=True``
    renvoient 400 ``invalid_json`` si le corps est vide ou mal formé. Cette
    fonction centralise un parsing tolérant pour les routes métier.
    """
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return cast(Dict[str, Any], data)
    if isinstance(data, str):
        try:
            nested = json.loads(data)
            if isinstance(nested, dict):
                return cast(Dict[str, Any], nested)
        except json.JSONDecodeError:
            pass

    raw = request.get_data(cache=True, as_text=True)
    if not raw or not raw.strip():
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if isinstance(parsed, dict):
        return cast(Dict[str, Any], parsed)
    if isinstance(parsed, str):
        try:
            nested = json.loads(parsed)
            if isinstance(nested, dict):
                return cast(Dict[str, Any], nested)
        except json.JSONDecodeError:
            pass
    return {}


def validate_request(
    schema: Schema, data: Dict[str, Any], strict: bool = True
) -> Dict[str, Any]:
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
        validated = schema.load(data, unknown="exclude" if strict else "include")
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

    Si _error_code et _details sont présents (ex: MATERIAL_DELIVERY_DESCRIPTION_REQUIRED),
    ajoute error, error_code, details pour harmoniser avec les erreurs API (A, C).

    Args:
        errors: Messages d'erreur bruts de Marshmallow

    Returns:
        Dict formaté avec structure standardisée
    """
    formatted: Dict[str, Any] = {
        "message": "Erreur de validation des données",
        "errors": {},
    }

    # Harmonisation B : error_code + details (contrat unifié avec A, C)
    error_code = errors.get("_error_code")
    details = errors.get("_details")
    if error_code and isinstance(details, dict):
        first_field = next(
            (k for k in errors if not k.startswith("_")),
            None,
        )
        first_msg = ""
        if first_field:
            msgs = errors[first_field]
            first_msg = msgs[0] if isinstance(msgs, list) and msgs else str(msgs)
        formatted["error"] = first_msg or "Erreur de validation"
        formatted["error_code"] = error_code
        formatted["details"] = details

    for field, messages in errors.items():
        # ⚡ Ignorer les clés spéciales de Marshmallow (_schema, _nested, etc.)
        # qui ne sont pas des champs de formulaire
        if field.startswith("_"):
            # Si c'est une erreur au niveau du schéma, l'ajouter au message général
            if field == "_schema" and isinstance(messages, list):
                formatted["message"] = (
                    messages[0] if messages else "Erreur de validation des données"
                )
            continue

        # Si c'est une liste de messages, prendre directement
        if isinstance(messages, list):
            formatted["errors"][field] = messages
        # Si c'est un dict (champs nested ou erreurs de liste),
        # formater récursivement
        elif isinstance(messages, dict):
            # ⚡ Détecter si c'est une erreur de validation de liste
            # (clés = indices entiers)
            # Exemple: {'client_ids': {1: ['Must be greater than or equal to 1.']}}
            all_keys_are_int_indices = all(
                (isinstance(k, int) or (isinstance(k, str) and k.isdigit()))
                for k in messages
            )

            if all_keys_are_int_indices:
                # ⚡ Cas: erreur de validation de liste
                # (regrouper toutes les erreurs sous le nom du champ)
                all_list_errors: list[str] = []
                for _, index_msgs in messages.items():
                    if isinstance(index_msgs, list):
                        all_list_errors.extend(index_msgs)
                    else:
                        all_list_errors.append(str(index_msgs))
                formatted["errors"][field] = all_list_errors
            # ⚡ Éviter de créer une structure errors.errors.errors...
            # Si le dict contient déjà "errors" ou "message",
            # extraire directement les champs
            elif "errors" in messages and isinstance(messages["errors"], dict):
                # Cas: erreur nested avec structure {errors: {...}}
                for nested_field, nested_msgs in messages["errors"].items():
                    if isinstance(nested_msgs, list):
                        formatted["errors"][f"{field}.{nested_field}"] = nested_msgs
                    else:
                        formatted["errors"][f"{field}.{nested_field}"] = [
                            str(nested_msgs)
                        ]
            else:
                # Formatage normal récursif pour champs nested
                nested_formatted = _format_validation_errors(
                    cast(Dict[str, Any], messages)
                )
                # Fusionner les erreurs nested directement dans formatted["errors"]
                if "errors" in nested_formatted:
                    for nested_field, nested_msgs in nested_formatted["errors"].items():
                        formatted["errors"][f"{field}.{nested_field}"] = nested_msgs
        # Sinon, convertir en liste
        else:
            formatted["errors"][field] = [str(messages)]

    return formatted


def validate_query_params(
    schema: Schema, query_params: Any, strict: bool = False
) -> Dict[str, Any]:
    """Valide les query parameters GET avec un schema Marshmallow.

    Args:
        schema: Schema Marshmallow à utiliser pour la validation
        query_params: Query parameters à valider
            (dict, request.args ou ImmutableMultiDict)
        strict: Si True, rejette les champs inconnus
            (défaut: False pour query params)

    Returns:
        Dict validé et nettoyé

    Raises:
        ValidationError: Si la validation échoue (avec détails par champ)

    Usage:
        from flask import request
        from schemas.query_schemas import PaginationQuerySchema
        from schemas.validation_utils import validate_query_params

        try:
            validated_params = validate_query_params(
                PaginationQuerySchema(), request.args
            )
            page = validated_params.get("page", 1)
            per_page = validated_params.get("per_page", 50)
        except ValidationError as e:
            return handle_validation_error(e)
    """
    # Convertir request.args (ImmutableMultiDict) en dict normal
    if hasattr(query_params, "to_dict"):
        # Flask request.args a une méthode to_dict(flat=True)
        data = query_params.to_dict(flat=True)
    elif hasattr(query_params, "dict"):
        # Compatible avec d'autres types de MultiDict
        data = query_params.dict()
    elif isinstance(query_params, dict):
        data = query_params
    else:
        # Fallback: convertir en dict
        data = dict(query_params)

    # Valider avec le schéma (même logique que validate_request)
    try:
        validated = schema.load(data, unknown="exclude" if strict else "include")
        return cast(Dict[str, Any], validated)
    except ValidationError as err:
        formatted_errors = _format_validation_errors(cast(Dict[str, Any], err.messages))
        raise ValidationError(formatted_errors) from err


def handle_validation_error(error: ValidationError):
    """Gère une ValidationError et retourne une réponse Flask 400.

    ✅ Format standard v2:
    {
        "error": "validation_error",
        "message": "Premier message d'erreur",
        "details": {"errors": {...}}
    }

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
    messages = cast(Dict[str, Any], error.messages)

    # Déjà formaté par validate_request (error_code présent) → retourner tel quel
    if "error_code" in messages:
        return messages, 400

    # Les validateurs internes remontent déjà parfois la structure
    # ``{"message": ..., "errors": {...}}``. La déplier évite des clés
    # artificielles telles que ``errors.for_date``.
    first_message = "Erreur de validation des données"
    errors_dict: Dict[str, Any] = {}

    formatted_errors = messages.get("errors")
    if isinstance(formatted_errors, dict):
        errors_dict = formatted_errors
        message = messages.get("message")
        if isinstance(message, str) and message:
            first_message = message
    else:
        for field, field_errors in messages.items():
            if field.startswith("_"):
                continue  # Ignorer les champs internes

            if isinstance(field_errors, list) and field_errors:
                errors_dict[field] = field_errors
                if first_message == "Erreur de validation des données":
                    first_message = field_errors[0]
            elif isinstance(field_errors, str):
                errors_dict[field] = [field_errors]
                if first_message == "Erreur de validation des données":
                    first_message = field_errors
            elif isinstance(field_errors, dict):
                # Champs imbriqués
                for subfield, suberrors in field_errors.items():
                    key = f"{field}.{subfield}"
                    if isinstance(suberrors, list) and suberrors:
                        errors_dict[key] = suberrors
                        if first_message == "Erreur de validation des données":
                            first_message = suberrors[0]

    return {
        "error": "validation_error",
        "message": first_message,
        # Compatibilité avec les consommateurs du format v1.
        "errors": errors_dict,
        "details": {"fields": errors_dict} if errors_dict else None,
    }, 400


# Validators personnalisés réutilisables
EMAIL_VALIDATOR = Length(
    min=5, max=254, error="Email doit faire entre 5 et 254 caractères"
)
USERNAME_VALIDATOR = Length(
    min=3, max=50, error="Username doit faire entre 3 et 50 caractères"
)
PASSWORD_VALIDATOR = Length(
    min=8, error="Mot de passe doit faire au moins 8 caractères"
)
PHONE_VALIDATOR = Length(
    min=10, max=20, error="Téléphone doit faire entre 10 et 20 caractères"
)

# Formats de validation courants
ISO8601_DATE_REGEX = r"^\d{4}-\d{2}-\d{2}$"
ISO8601_DATETIME_REGEX = (
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?(Z|[+-]\d{2}:\d{2})?$"
)
