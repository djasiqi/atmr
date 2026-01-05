"""Template de routes standardisées pour l'architecture DDD.

Ce fichier contient des exemples de templates pour créer des routes
qui suivent le pattern standardisé de l'application.

Pattern standardisé :
1. Validation des inputs (si nécessaire)
2. Exécution du use case
3. Gestion du résultat (erreurs, not found, etc.)
4. Retour d'une réponse standardisée

Note: Ce fichier est un template de référence, ne pas l'importer directement.
"""

import logging

from flask import Blueprint, request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Namespace, Resource  # pyright: ignore[reportMissingImports]

from shared.error_handlers import APIErrorHandler
from shared.response_helpers import (
    created_response,
    paginated_response,
    success_response,
)

logger = logging.getLogger(__name__)

# Constantes pour éviter les valeurs magiques
MAX_PER_PAGE = 100

# ============================================================================
# TEMPLATE 1: Route Flask-RESTx avec Use Case (GET)
# ============================================================================

# Namespace Flask-RESTx
example_ns = Namespace("example", description="Exemple de routes")

# Modèle Swagger (optionnel, pour documentation)
example_model = example_ns.model(
    "Example",
    {
        "id": example_ns.fields.Integer(required=True),
        "name": example_ns.fields.String(required=True),
    },
)


@example_ns.route("/<int:example_id>")
class ExampleResource(Resource):
    """Exemple de route GET avec use case."""

    @jwt_required()
    @example_ns.doc(responses={200: "Succès", 404: "Non trouvé", 500: "Erreur serveur"})
    @example_ns.marshal_with(example_model)
    def get(self, example_id: int):
        """Récupère un exemple par son ID."""
        try:
            # 1. Validation (si nécessaire)
            if example_id <= 0:
                return APIErrorHandler.handle_validation_error(
                    "example_id must be positive",
                    field="example_id",
                    logger_instance=logger,
                )

            # 2. Exécuter use case
            # from application.examples.get_example import GetExampleUseCase, GetExampleInput
            # uc = GetExampleUseCase(example_repo=example_repo)
            # input_data = GetExampleInput(example_id=example_id)
            # result = uc.execute(input_data)

            # 3. Gérer le résultat
            # if not result.success:
            #     return APIErrorHandler.handle_validation_error(
            #         result.error.get("message", "Erreur inconnue") if result.error else "Erreur inconnue",
            #         logger_instance=logger,
            #     )

            # if not result.example:
            #     return APIErrorHandler.handle_not_found(
            #         "Example",
            #         resource_id=example_id,
            #         logger_instance=logger,
            #     )

            # 4. Retourner réponse de succès
            # return success_response(data=result.example.to_dict())

            # Exemple de réponse (à remplacer par le code ci-dessus)
            return success_response(data={"id": example_id, "name": "Example"})

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


# ============================================================================
# TEMPLATE 2: Route Flask-RESTx avec Use Case (POST)
# ============================================================================


@example_ns.route("")
class CreateExample(Resource):
    """Exemple de route POST avec use case."""

    @jwt_required()
    @example_ns.expect(example_model)  # Pour Swagger
    @example_ns.doc(
        responses={201: "Créé", 400: "Erreur de validation", 500: "Erreur serveur"}
    )
    def post(self):
        """Crée un nouvel exemple."""
        try:
            # 1. Extraire et valider les données
            json_data = request.get_json()
            if not json_data:
                return APIErrorHandler.handle_validation_error(
                    "JSON body is required",
                    logger_instance=logger,
                )

            # 2. Créer l'input du use case
            # from application.examples.create_example import CreateExampleUseCase, CreateExampleInput
            # input_data = CreateExampleInput(
            #     name=json_data.get("name"),
            #     # ... autres champs
            # )

            # 3. Exécuter use case
            # uc = CreateExampleUseCase(example_repo=example_repo)
            # result = uc.execute(input_data)

            # 4. Gérer le résultat
            # if not result.success:
            #     return APIErrorHandler.handle_validation_error(
            #         result.error.get("message", "Erreur inconnue") if result.error else "Erreur inconnue",
            #         logger_instance=logger,
            #     )

            # 5. Retourner réponse 201
            # return created_response(
            #     data=result.example.to_dict(),
            #     location=f"/api/examples/{result.example.id}",
            # )

            # Exemple de réponse (à remplacer par le code ci-dessus)
            return created_response(
                data={"id": 1, "name": json_data.get("name", "Example")},
                location="/api/examples/1",
            )

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


# ============================================================================
# TEMPLATE 3: Route Flask-RESTx avec Use Case (List/Pagination)
# ============================================================================


@example_ns.route("")
class ListExamples(Resource):
    """Exemple de route GET avec pagination."""

    @jwt_required()
    @example_ns.doc(
        params={
            "page": "Numéro de page (défaut: 1)",
            "per_page": "Éléments par page (défaut: 50)",
        },
        responses={200: "Succès", 400: "Erreur de validation", 500: "Erreur serveur"},
    )
    def get(self):
        """Liste les exemples avec pagination."""
        try:
            # 1. Extraire les paramètres de pagination
            page = request.args.get("page", 1, type=int)
            per_page = request.args.get("per_page", 50, type=int)

            # Validation
            if page < 1:
                return APIErrorHandler.handle_validation_error(
                    "page must be >= 1",
                    field="page",
                    provided_value=page,
                    logger_instance=logger,
                )
            if per_page < 1 or per_page > MAX_PER_PAGE:
                return APIErrorHandler.handle_validation_error(
                    f"per_page must be between 1 and {MAX_PER_PAGE}",
                    field="per_page",
                    provided_value=per_page,
                    expected_format=f"1-{MAX_PER_PAGE}",
                    logger_instance=logger,
                )

            # 2. Exécuter use case
            # from application.examples.list_examples import ListExamplesUseCase, ListExamplesInput
            # uc = ListExamplesUseCase(example_repo=example_repo)
            # input_data = ListExamplesInput(page=page, per_page=per_page)
            # result = uc.execute(input_data)

            # 3. Gérer le résultat
            # if not result.success:
            #     return APIErrorHandler.handle_validation_error(
            #         result.error.get("message", "Erreur inconnue") if result.error else "Erreur inconnue",
            #         logger_instance=logger,
            #     )

            # 4. Construire les liens de pagination (optionnel)
            # links = {}
            # if result.total and result.examples:
            #     total_pages = (result.total + per_page - 1) // per_page
            #     if page < total_pages:
            #         links["next"] = f"/api/examples?page={page + 1}&per_page={per_page}"
            #     if page > 1:
            #         links["prev"] = f"/api/examples?page={page - 1}&per_page={per_page}"
            #     links["first"] = f"/api/examples?page=1&per_page={per_page}"
            #     links["last"] = f"/api/examples?page={total_pages}&per_page={per_page}"

            # 5. Retourner réponse paginée
            # return paginated_response(
            #     items=[e.to_dict() for e in result.examples],
            #     total=result.total or 0,
            #     page=page,
            #     per_page=per_page,
            #     links=links if links else None,
            # )

            # Exemple de réponse (à remplacer par le code ci-dessus)
            return paginated_response(
                items=[{"id": 1, "name": "Example 1"}, {"id": 2, "name": "Example 2"}],
                total=2,
                page=page,
                per_page=per_page,
            )

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


# ============================================================================
# TEMPLATE 4: Route Blueprint avec Use Case (GET)
# ============================================================================

# Blueprint
example_bp = Blueprint("example", __name__, url_prefix="/api/examples")


@example_bp.route("/<int:example_id>", methods=["GET"])
@jwt_required()
def get_example(example_id: int):
    """Récupère un exemple par son ID (Blueprint)."""
    try:
        # 1. Validation
        if example_id <= 0:
            return APIErrorHandler.handle_validation_error(
                "example_id must be positive",
                field="example_id",
                logger_instance=logger,
            )

        # 2. Exécuter use case
        # from application.examples.get_example import GetExampleUseCase, GetExampleInput
        # uc = GetExampleUseCase(example_repo=example_repo)
        # input_data = GetExampleInput(example_id=example_id)
        # result = uc.execute(input_data)

        # 3. Gérer le résultat
        # if not result.success:
        #     return APIErrorHandler.handle_validation_error(
        #         result.error.get("message", "Erreur inconnue") if result.error else "Erreur inconnue",
        #         logger_instance=logger,
        #     )

        # if not result.example:
        #     return APIErrorHandler.handle_not_found(
        #         "Example",
        #         resource_id=example_id,
        #         logger_instance=logger,
        #     )

        # 4. Retourner réponse
        # return success_response(data=result.example.to_dict())

        # Exemple de réponse (à remplacer par le code ci-dessus)
        return success_response(data={"id": example_id, "name": "Example"})

    except Exception as e:
        return APIErrorHandler.handle_exception(e, logger)


# ============================================================================
# TEMPLATE 5: Route Blueprint avec Use Case (POST)
# ============================================================================


@example_bp.route("", methods=["POST"])
@jwt_required()
def create_example():
    """Crée un nouvel exemple (Blueprint)."""
    try:
        # 1. Extraire et valider les données
        json_data = request.get_json()
        if not json_data:
            return APIErrorHandler.handle_validation_error(
                "JSON body is required",
                logger_instance=logger,
            )

        # 2. Créer l'input du use case
        # from application.examples.create_example import CreateExampleUseCase, CreateExampleInput
        # input_data = CreateExampleInput(
        #     name=json_data.get("name"),
        #     # ... autres champs
        # )

        # 3. Exécuter use case
        # uc = CreateExampleUseCase(example_repo=example_repo)
        # result = uc.execute(input_data)

        # 4. Gérer le résultat
        # if not result.success:
        #     return APIErrorHandler.handle_validation_error(
        #         result.error.get("message", "Erreur inconnue") if result.error else "Erreur inconnue",
        #         logger_instance=logger,
        #     )

        # 5. Retourner réponse 201
        # return created_response(
        #     data=result.example.to_dict(),
        #     location=f"/api/examples/{result.example.id}",
        # )

        # Exemple de réponse (à remplacer par le code ci-dessus)
        return created_response(
            data={"id": 1, "name": json_data.get("name", "Example")},
            location="/api/examples/1",
        )

    except Exception as e:
        return APIErrorHandler.handle_exception(e, logger)


# ============================================================================
# TEMPLATE 6: Route Flask-RESTx sans Use Case (Legacy)
# ============================================================================

# Pour les routes qui n'ont pas encore été migrées vers des use cases,
# utiliser ce pattern temporairement :


@example_ns.route("/legacy/<int:example_id>")
class LegacyExampleResource(Resource):
    """Exemple de route legacy (sans use case)."""

    @jwt_required()
    def get(self, example_id: int):
        """Récupère un exemple (legacy, sans use case)."""
        try:
            # 1. Validation
            if example_id <= 0:
                return APIErrorHandler.handle_validation_error(
                    "example_id must be positive",
                    field="example_id",
                    logger_instance=logger,
                )

            # 2. Logique métier directe (à migrer vers un use case)
            # example = example_repo.find_by_id(example_id)
            # if not example:
            #     return APIErrorHandler.handle_not_found(
            #         "Example",
            #         resource_id=example_id,
            #         logger_instance=logger,
            #     )

            # 3. Retourner réponse
            # return success_response(data=example.to_dict())

            # Exemple de réponse (à remplacer par le code ci-dessus)
            return success_response(data={"id": example_id, "name": "Example"})

        except Exception as e:
            return APIErrorHandler.handle_exception(e, logger)


# ============================================================================
# TEMPLATE 7: Route Blueprint sans Use Case (Legacy)
# ============================================================================


@example_bp.route("/legacy/<int:example_id>", methods=["GET"])
@jwt_required()
def get_legacy_example(example_id: int):
    """Récupère un exemple (legacy, sans use case)."""
    try:
        # 1. Validation
        if example_id <= 0:
            return APIErrorHandler.handle_validation_error(
                "example_id must be positive",
                field="example_id",
                logger_instance=logger,
            )

        # 2. Logique métier directe (à migrer vers un use case)
        # example = example_repo.find_by_id(example_id)
        # if not example:
        #     return APIErrorHandler.handle_not_found(
        #         "Example",
        #         resource_id=example_id,
        #         logger_instance=logger,
        #     )

        # 3. Retourner réponse
        # return success_response(data=example.to_dict())

        # Exemple de réponse (à remplacer par le code ci-dessus)
        return success_response(data={"id": example_id, "name": "Example"})

    except Exception as e:
        return APIErrorHandler.handle_exception(e, logger)
