"""Routes API pour le monitoring des rotations de secrets.

Endpoints:
    GET  /api/v1/admin/secret-rotations/history  - Historique des rotations
    GET  /api/v1/admin/secret-rotations/stats     - Statistiques globales
    GET  /api/v1/admin/secret-rotations/last     - Dernière rotation par type
"""

import logging

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource, fields

from ext import role_required
from models import UserRole
from services.secret_rotation_monitor import (
    get_days_since_last_rotation,
    get_last_rotation,
    get_rotation_history,
    get_rotation_stats,
)

logger = logging.getLogger(__name__)

# Créer le namespace
secret_rotation_ns = Namespace(
    "secret-rotations",
    description="Monitoring des rotations de secrets via Vault",
    path="/admin/secret-rotations",
)

# Modèles Swagger pour les réponses
rotation_item_model = secret_rotation_ns.model(
    "RotationItem",
    {
        "id": fields.Integer(description="ID de la rotation"),
        "secret_type": fields.String(description="Type de secret (jwt, encryption, flask_secret_key)"),
        "status": fields.String(description="Statut (success, error, skipped)"),
        "rotated_at": fields.String(description="Date de rotation (ISO 8601)"),
        "environment": fields.String(description="Environnement (dev, prod, testing)"),
        "metadata": fields.Raw(description="Métadonnées additionnelles"),
        "error_message": fields.String(description="Message d'erreur si status=error"),
        "task_id": fields.String(description="ID de la tâche Celery"),
    },
)

rotation_history_model = secret_rotation_ns.model(
    "RotationHistory",
    {
        "rotations": fields.List(fields.Nested(rotation_item_model), description="Liste des rotations"),
        "total": fields.Integer(description="Nombre total de rotations"),
        "page": fields.Integer(description="Page actuelle (basé sur offset)"),
        "per_page": fields.Integer(description="Nombre d'éléments par page"),
    },
)

rotation_stats_model = secret_rotation_ns.model(
    "RotationStats",
    {
        "total_rotations": fields.Integer(description="Nombre total de rotations"),
        "success_count": fields.Integer(description="Nombre de rotations réussies"),
        "error_count": fields.Integer(description="Nombre d'erreurs"),
        "skipped_count": fields.Integer(description="Nombre de rotations ignorées"),
        "by_type": fields.Raw(description="Statistiques par type de secret (jwt, encryption, flask_secret_key)"),
        "last_rotations": fields.Raw(description="Date de dernière rotation par type (ISO 8601)"),
    },
)

last_rotation_model = secret_rotation_ns.model(
    "LastRotation",
    {
        "secret_type": fields.String(description="Type de secret"),
        "rotation": fields.Nested(rotation_item_model, allow_null=True, description="Dernière rotation ou null"),
        "days_since_last": fields.Integer(
            allow_null=True, description="Nombre de jours depuis la dernière rotation réussie"
        ),
    },
)


@secret_rotation_ns.route("/history")
class RotationHistory(Resource):
    @secret_rotation_ns.doc(
        description="Récupère l'historique des rotations de secrets avec filtres optionnels et pagination.",
        params={
            "secret_type": "Filtrer par type de secret (jwt, encryption, flask_secret_key)",
            "status": "Filtrer par statut (success, error, skipped)",
            "environment": "Filtrer par environnement (dev, prod, testing)",
            "limit": "Nombre maximum de résultats (défaut: 50, max: 100)",
            "offset": "Offset pour pagination (défaut: 0)",
        },
    )
    @secret_rotation_ns.response(200, "Succès", rotation_history_model)
    @secret_rotation_ns.response(401, "Non autorisé")
    @secret_rotation_ns.response(403, "Accès refusé (admin uniquement)")
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère l'historique des rotations de secrets."""
        try:
            # Récupérer les paramètres de requête
            secret_type = request.args.get("secret_type", type=str)
            status = request.args.get("status", type=str)
            environment = request.args.get("environment", type=str)
            limit = min(request.args.get("limit", 50, type=int), 100)  # Max 100
            offset = max(request.args.get("offset", 0, type=int), 0)

            # Récupérer l'historique
            rotations, total = get_rotation_history(
                secret_type=secret_type,
                status=status,
                environment=environment,
                limit=limit,
                offset=offset,
            )

            # Calculer la page (approximative basée sur offset)
            page = (offset // limit) + 1 if limit > 0 else 1

            return {
                "rotations": [r.to_dict() for r in rotations],
                "total": total,
                "page": page,
                "per_page": limit,
            }, 200

        except Exception as e:
            logger.exception("[SecretRotationAPI] Erreur récupération historique: %s", e)
            return {"error": "Erreur lors de la récupération de l'historique", "details": str(e)}, 500


@secret_rotation_ns.route("/stats")
class RotationStats(Resource):
    @secret_rotation_ns.doc(
        description="Récupère les statistiques globales des rotations de secrets.",
    )
    @secret_rotation_ns.response(200, "Succès", rotation_stats_model)
    @secret_rotation_ns.response(401, "Non autorisé")
    @secret_rotation_ns.response(403, "Accès refusé (admin uniquement)")
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère les statistiques globales des rotations."""
        try:
            stats = get_rotation_stats()
            return stats, 200

        except Exception as e:
            logger.exception("[SecretRotationAPI] Erreur récupération statistiques: %s", e)
            return {"error": "Erreur lors de la récupération des statistiques", "details": str(e)}, 500


@secret_rotation_ns.route("/last")
class LastRotation(Resource):
    @secret_rotation_ns.doc(
        description="Récupère la dernière rotation pour chaque type de secret ou un type spécifique.",
        params={
            "secret_type": "Type de secret spécifique (jwt, encryption, flask_secret_key). Si omis, retourne tous les types.",
            "environment": "Filtrer par environnement (dev, prod, testing)",
        },
    )
    @secret_rotation_ns.response(200, "Succès", fields.List(fields.Nested(last_rotation_model)))
    @secret_rotation_ns.response(401, "Non autorisé")
    @secret_rotation_ns.response(403, "Accès refusé (admin uniquement)")
    @jwt_required()
    @role_required(UserRole.admin)
    def get(self):
        """Récupère la dernière rotation par type de secret."""
        try:
            secret_type = request.args.get("secret_type", type=str)
            environment = request.args.get("environment", type=str)

            if secret_type:
                # Retourner pour un type spécifique
                last = get_last_rotation(secret_type, environment=environment)
                days_since = get_days_since_last_rotation(secret_type, environment=environment)

                return [
                    {
                        "secret_type": secret_type,
                        "rotation": last.to_dict() if last else None,
                        "days_since_last": days_since,
                    }
                ], 200

            # Retourner pour tous les types
            result = []
            for st in ["jwt", "encryption", "flask_secret_key"]:
                last = get_last_rotation(st, environment=environment)
                days_since = get_days_since_last_rotation(st, environment=environment)

                result.append(
                    {
                        "secret_type": st,
                        "rotation": last.to_dict() if last else None,
                        "days_since_last": days_since,
                    }
                )

            return result, 200

        except Exception as e:
            logger.exception("[SecretRotationAPI] Erreur récupération dernière rotation: %s", e)
            return {"error": "Erreur lors de la récupération de la dernière rotation", "details": str(e)}, 500
