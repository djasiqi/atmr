# routes/app_version.py
"""Routes API pour la vérification de version de l'application mobile.

Endpoints:
    POST /api/v1/app/version-check - Vérifie si une mise à jour est requise/recommandée
"""

from __future__ import annotations

import logging
from typing import Any

from flask import request
from flask_restx import Namespace, Resource, fields
from werkzeug.exceptions import BadRequest

from services.version_check import check_app_version

logger = logging.getLogger(__name__)

app_version_ns = Namespace(
    "app", description="Endpoints pour la gestion des versions de l'application mobile"
)

# Modèles de validation pour Swagger/Flask-RESTX
version_check_request = app_version_ns.model(
    "VersionCheckRequest",
    {
        "platform": fields.String(
            required=True,
            description="Plateforme: 'android' ou 'ios'",
            example="android",
        ),
        "current_version": fields.String(
            required=True,
            description="Version actuelle de l'application (format semver: MAJOR.MINOR.PATCH)",
            example="1.2.3",
        ),
    },
)

version_check_response = app_version_ns.model(
    "VersionCheckResponse",
    {
        "platform": fields.String(description="Plateforme", example="android"),
        "current_version": fields.String(
            description="Version actuelle", example="1.2.3"
        ),
        "latest_version": fields.String(
            description="Dernière version disponible", example="1.3.0"
        ),
        "min_required_version": fields.String(
            description="Version minimale requise", example="1.2.0"
        ),
        "status": fields.String(
            description="Statut: OK, UPDATE_RECOMMENDED, ou UPDATE_REQUIRED",
            example="UPDATE_RECOMMENDED",
        ),
        "store_url": fields.String(
            description="URL du store pour la mise à jour",
            example="https://play.google.com/store/apps/details?id=com.drinjasiqi.atmr",
            allow_null=True,
        ),
        "message": fields.String(
            description="Message personnalisé pour la mise à jour",
            example="Une nouvelle version est disponible...",
            allow_null=True,
        ),
    },
)


@app_version_ns.route("/version-check")
class VersionCheck(Resource):
    """Endpoint pour vérifier si une mise à jour est requise ou recommandée."""

    @app_version_ns.expect(version_check_request)
    @app_version_ns.marshal_with(version_check_response)
    @app_version_ns.doc(
        description=(
            "Vérifie la version de l'application et retourne le statut de mise à jour.\n\n"
            "**Statuts possibles:**\n"
            "- `OK`: L'application est à jour\n"
            "- `UPDATE_RECOMMENDED`: Une mise à jour est recommandée (non bloquante)\n"
            "- `UPDATE_REQUIRED`: Une mise à jour est obligatoire (bloquante)\n\n"
            "**Logique:**\n"
            "- Si `current_version < min_required_version` → `UPDATE_REQUIRED`\n"
            "- Sinon si `current_version < latest_version` → `UPDATE_RECOMMENDED`\n"
            "- Sinon → `OK`\n\n"
            "**Note:** Cet endpoint est public (pas d'authentification requise) car il doit "
            "être appelé avant même que l'utilisateur soit connecté."
        ),
        responses={
            200: "Vérification réussie",
            400: "Requête invalide (plateforme ou version invalide)",
            500: "Erreur serveur",
        },
    )
    def post(self) -> tuple[dict[str, Any], int]:
        """Vérifie la version de l'application."""
        try:
            data = request.get_json() or {}
            platform = data.get("platform", "").strip().lower()
            current_version = data.get("current_version", "").strip()

            # Validation
            if not platform:
                raise BadRequest("Le champ 'platform' est requis (android ou ios)")
            if platform not in ("android", "ios"):
                raise BadRequest(
                    f"Plateforme invalide: {platform}. Attendu: 'android' ou 'ios'"
                )
            if not current_version:
                raise BadRequest("Le champ 'current_version' est requis")

            # Vérification de version
            result = check_app_version(platform, current_version)

            logger.info(
                "Version check: %s %s → %s",
                platform,
                current_version,
                result["status"],
            )

            return result, 200

        except BadRequest as e:
            logger.warning("Version check - requête invalide: %s", e)
            return {"error": str(e)}, 400
        except ValueError as e:
            logger.warning("Version check - erreur de validation: %s", e)
            return {"error": str(e)}, 400
        except Exception as e:
            logger.exception("Version check - erreur serveur: %s", e)
            return {"error": "Erreur serveur lors de la vérification de version"}, 500
