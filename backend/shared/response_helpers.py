"""Helpers centralisés pour créer des réponses JSON standardisées.

Ce module fournit des fonctions helper pour uniformiser les réponses JSON
dans toutes les routes, améliorant la cohérence et la maintenabilité.
"""

from typing import Any

from flask import Response, jsonify, make_response


def json_response(
    payload: dict[str, Any] | list[Any],
    status_code: int = 200,
) -> Response:
    """Réponse HTTP JSON explicite (application/json).

    À utiliser à la place de ``make_response(dict)`` lorsqu'on doit encore
    poser des cookies ou des en-têtes sur l'objet Response.
    """
    return make_response(jsonify(payload), status_code)


def rewrap_as_json_response(inner: Any) -> Any:
    """Reconstruit une Response Flask via jsonify, en recopiant les cookies.

    Sert aux wrappers compat qui délèguent à une Resource RESTX : le contrat
    HTTP reste identique, le Content-Type est explicitement application/json.
    """
    if isinstance(inner, tuple):
        body = inner[0]
        status = int(inner[1]) if len(inner) > 1 else 200
        if isinstance(body, (dict, list)):
            return json_response(body, status)
        return inner

    payload = None
    get_json = getattr(inner, "get_json", None)
    if callable(get_json):
        payload = get_json(silent=True)
    if not isinstance(payload, (dict, list)):
        return inner

    status = int(getattr(inner, "status_code", 200) or 200)
    out = json_response(payload, status)
    headers = getattr(inner, "headers", None)
    if headers is not None and hasattr(headers, "getlist"):
        for cookie_header in headers.getlist("Set-Cookie"):
            out.headers.add("Set-Cookie", cookie_header)
    return out


def success_response(
    data: dict[str, Any] | list[Any] | None = None,
    status_code: int = 200,
    message: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Crée une réponse de succès standardisée.

    Args:
        data: Données à retourner (dict, list, ou None)
        status_code: Code HTTP de statut (défaut: 200)
        message: Message optionnel à inclure

    Returns:
        Tuple (response_json, status_code) pour Flask

    Examples:
        # Réponse simple avec données
        return success_response(data={"id": 1, "name": "Test"})

        # Réponse avec message
        return success_response(
            data={"id": 1},
            message="Opération réussie",
            status_code=200
        )

        # Réponse sans données (juste message)
        return success_response(message="Opération réussie")
    """
    response: dict[str, Any] = {}
    if message:
        response["message"] = message
    if data is not None:
        response["data"] = data
    return response, status_code


def created_response(
    data: dict[str, Any] | list[Any] | None = None,
    location: str | None = None,  # noqa: ARG001
    message: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Crée une réponse 201 (Created) standardisée.

    Args:
        data: Données de la ressource créée
        location: URL de la ressource créée (pour header Location) - non
            utilisé actuellement, le header Location doit être ajouté
            séparément dans la route si nécessaire
        message: Message optionnel

    Returns:
        Tuple (response_json, status_code) pour Flask

    Examples:
        return created_response(
            data={"id": 1, "name": "Test"},
            location="/api/payments/1"
        )
    """
    response: dict[str, Any] = {}
    if message:
        response["message"] = message
    if data is not None:
        response["data"] = data
    # Note: Le header Location doit être ajouté séparément dans la route
    # Flask-RESTx le gère automatiquement si on retourne un tuple avec location
    return response, 201


def no_content_response() -> tuple[dict[str, Any], int]:
    """Crée une réponse 204 (No Content) standardisée.

    Returns:
        Tuple (response_json, status_code) pour Flask

    Examples:
        return no_content_response()
    """
    return {}, 204


def paginated_response(
    items: list[Any],
    total: int,
    page: int,
    per_page: int,
    links: dict[str, str] | None = None,
    message: str | None = None,
) -> tuple[dict[str, Any], int]:
    """Crée une réponse paginée standardisée.

    Args:
        items: Liste des éléments de la page actuelle
        total: Nombre total d'éléments
        page: Numéro de page actuelle (1-indexed)
        per_page: Nombre d'éléments par page
        links: Liens de pagination (prev, next, first, last) - optionnel
        message: Message optionnel

    Returns:
        Tuple (response_json, status_code) pour Flask

    Examples:
        return paginated_response(
            items=[{"id": 1}, {"id": 2}],
            total=100,
            page=1,
            per_page=10,
            links={
                "next": "/api/payments?page=2",
                "last": "/api/payments?page=10"
            }
        )
    """
    pages = (total + per_page - 1) // per_page if total > 0 else 0
    response: dict[str, Any] = {
        "data": items,
        "pagination": {
            "total": total,
            "page": page,
            "per_page": per_page,
            "pages": pages,
            "has_next": page < pages,
            "has_prev": page > 1,
        },
    }
    if message:
        response["message"] = message
    if links:
        response["links"] = links
    return response, 200
