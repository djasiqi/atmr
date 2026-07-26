# Constantes pour éviter les valeurs magiques
import logging
from typing import Any

from flask import Blueprint, jsonify, request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from werkzeug.exceptions import BadRequest  # pyright: ignore[reportMissingImports]

from ext import role_required
from models import UserRole
from security.ip_whitelist import ip_whitelist_required
from services.infrastructure.feature_flags import FeatureFlags, get_feature_flags_status
from services.infrastructure.runtime_flags import get_runtime_flags_status

PERCENTAGE_PERCENT = 100
MIN_SUCCESS_RATE = 0.95
MIN_REQUESTS_FOR_ALERT = 10
MAX_FAILURES_THRESHOLD = 50

"""Routes API pour gérer les feature flags (F-04 : ADMIN + IP uniquement).

Endpoints:
    GET  /api/feature-flags/status         - Statut actuel
    POST /api/feature-flags/ml/enable      - Activer ML
    POST /api/feature-flags/ml/disable     - Désactiver ML
    POST /api/feature-flags/ml/percentage  - Modifier pourcentage
    POST /api/feature-flags/reset-stats    - Reset stats
"""
# pyright: reportReturnType=false
# Flask jsonify() retourne Response, pas dict


logger = logging.getLogger(__name__)

# Créer le blueprint
feature_flags_bp = Blueprint("feature_flags", __name__, url_prefix="/api/feature-flags")


@feature_flags_bp.route("/status", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_status() -> tuple[dict[str, Any], int]:
    """Récupère le statut actuel des feature flags."""
    try:
        status = get_feature_flags_status()
        return jsonify(status), 200

    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)


@feature_flags_bp.route("/runtime-status", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_runtime_status() -> tuple[dict[str, Any], int]:
    """Statut des flags runtime (Socket.IO, relay, kill-switch startup iOS)."""
    try:
        return jsonify(get_runtime_flags_status()), 200
    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)


@feature_flags_bp.route("/ml/enable", methods=["POST"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def enable_ml() -> tuple[dict[str, Any], int]:
    """Active le ML."""
    try:
        data = request.get_json() or {}
        percentage = data.get("percentage", 10)

        if not isinstance(percentage, int) or not 0 <= percentage <= PERCENTAGE_PERCENT:
            msg = "Percentage must be an integer between 0 and 100"
            raise BadRequest(msg)

        FeatureFlags.set_ml_enabled(True)
        FeatureFlags.set_ml_traffic_percentage(percentage)

        logger.warning("[FeatureFlagsAPI] ML enabled at %s%% via API", percentage)

        return jsonify(
            {
                "success": True,
                "message": f"ML activé à {percentage}%",
                "status": get_feature_flags_status(),
            }
        ), 200

    except BadRequest as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_validation_error(str(e), logger_instance=logger)
    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)


@feature_flags_bp.route("/ml/disable", methods=["POST"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def disable_ml() -> tuple[dict[str, Any], int]:
    """Désactive le ML."""
    try:
        FeatureFlags.set_ml_enabled(False)
        FeatureFlags.set_ml_traffic_percentage(0)

        logger.warning("[FeatureFlagsAPI] ML disabled via API")

        return jsonify(
            {
                "success": True,
                "message": "ML désactivé",
                "status": get_feature_flags_status(),
            }
        ), 200

    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)


@feature_flags_bp.route("/ml/percentage", methods=["POST"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def set_percentage() -> tuple[dict[str, Any], int]:
    """Modifie le pourcentage de trafic ML."""
    try:
        data = request.get_json()
        if not data or "percentage" not in data:
            msg = "'percentage' field required in request body"
            raise BadRequest(msg)

        percentage = data["percentage"]

        if not isinstance(percentage, int) or not 0 <= percentage <= PERCENTAGE_PERCENT:
            msg = "Percentage must be an integer between 0 and 100"
            raise BadRequest(msg)

        FeatureFlags.set_ml_traffic_percentage(percentage)

        logger.warning(
            "[FeatureFlagsAPI] ML percentage set to %s%% via API", percentage
        )

        return jsonify(
            {
                "success": True,
                "message": f"Trafic ML configuré à {percentage}%",
                "status": get_feature_flags_status(),
            }
        ), 200

    except BadRequest as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_validation_error(str(e), logger_instance=logger)
    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)


@feature_flags_bp.route("/reset-stats", methods=["POST"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def reset_stats() -> tuple[dict[str, Any], int]:
    """Réinitialise les statistiques."""
    try:
        FeatureFlags.reset_stats()

        logger.info("[FeatureFlagsAPI] Stats reset via API")

        return jsonify(
            {
                "success": True,
                "message": "Statistiques réinitialisées",
                "status": get_feature_flags_status(),
            }
        ), 200

    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)


@feature_flags_bp.route("/ml/health", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def ml_health() -> tuple[dict[str, Any], int]:
    """Vérifie la santé du système ML."""
    try:
        status = get_feature_flags_status()
        stats = status["stats"]

        is_healthy = (
            status["config"]["ML_ENABLED"]
            and stats["ml_requests"] > 0
            and stats["ml_success_rate"] >= MIN_SUCCESS_RATE
        )

        health_status = "healthy" if is_healthy else "degraded"

        alerts = []
        if (
            stats["ml_success_rate"] < MIN_SUCCESS_RATE
            and stats["ml_requests"] > MIN_REQUESTS_FOR_ALERT
        ):
            alerts.append(f"Taux de succès bas: {stats['ml_success_rate']:.2%}")

        if stats["ml_failures"] > MAX_FAILURES_THRESHOLD:
            alerts.append(f"Trop d'erreurs: {stats['ml_failures']}")

        return jsonify(
            {
                "status": health_status,
                "healthy": is_healthy,
                "ml_enabled": status["config"]["ML_ENABLED"],
                "ml_traffic": status["config"]["ML_TRAFFIC_PERCENTAGE"],
                "success_rate": stats["ml_success_rate"],
                "total_requests": stats["total_requests"],
                "alerts": alerts,
            }
        ), 200 if is_healthy else 503

    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        error_response, status_code = APIErrorHandler.handle_exception(e, logger)
        error_response["status"] = "error"
        error_response["healthy"] = False
        return jsonify(error_response), status_code
