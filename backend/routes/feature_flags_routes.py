"""
Routes API pour gérer les feature flags.

Endpoints:
    GET  /api/feature-flags/status         - Statut actuel
    POST /api/feature-flags/ml/enable      - Activer ML
    POST /api/feature-flags/ml/disable     - Désactiver ML
    POST /api/feature-flags/ml/percentage  - Modifier pourcentage
    POST /api/feature-flags/reset-stats    - Reset stats
"""
# ruff: noqa: ANN201
# pyright: reportReturnType=false
# Flask jsonify() retourne Response, pas dict
import logging
from typing import Any

from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest

from feature_flags import FeatureFlags, get_feature_flags_status

logger = logging.getLogger(__name__)

# Créer le blueprint
feature_flags_bp = Blueprint("feature_flags", __name__, url_prefix="/api/feature-flags")


@feature_flags_bp.route("/status", methods=["GET"])
def get_status() -> tuple[dict[str, Any], int]:
    """
    Récupère le statut actuel des feature flags.
    Returns:
        JSON avec configuration, stats et santé
    """
    try:
        status = get_feature_flags_status()
        return jsonify(status), 200

    except Exception as e:
        logger.error(f"[FeatureFlagsAPI] Error getting status: {e}")
        return jsonify({"error": str(e)}), 500


@feature_flags_bp.route("/ml/enable", methods=["POST"])
def enable_ml() -> tuple[dict[str, Any], int]:
    """
    Active le ML.
    Body (optionnel):
        {
            "percentage": 10  // 0-100, défaut: 10
        }
    Returns:
        JSON avec nouveau statut
    """
    try:
        data = request.get_json() or {}
        percentage = data.get("percentage", 10)

        if not isinstance(percentage, int) or not 0 <= percentage <= 100:
            raise BadRequest("Percentage must be an integer between 0 and 100")

        FeatureFlags.set_ml_enabled(True)
        FeatureFlags.set_ml_traffic_percentage(percentage)

        logger.warning(f"[FeatureFlagsAPI] ML enabled at {percentage}% via API")

        return jsonify({
            "success": True,
            "message": f"ML activé à {percentage}%",
            "status": get_feature_flags_status(),
        }), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"[FeatureFlagsAPI] Error enabling ML: {e}")
        return jsonify({"error": str(e)}), 500


@feature_flags_bp.route("/ml/disable", methods=["POST"])
def disable_ml() -> tuple[dict[str, Any], int]:
    """
    Désactive le ML.
    Returns:
        JSON avec nouveau statut
    """
    try:
        FeatureFlags.set_ml_enabled(False)
        FeatureFlags.set_ml_traffic_percentage(0)

        logger.warning("[FeatureFlagsAPI] ML disabled via API")

        return jsonify({
            "success": True,
            "message": "ML désactivé",
            "status": get_feature_flags_status(),
        }), 200

    except Exception as e:
        logger.error(f"[FeatureFlagsAPI] Error disabling ML: {e}")
        return jsonify({"error": str(e)}), 500


@feature_flags_bp.route("/ml/percentage", methods=["POST"])
def set_percentage() -> tuple[dict[str, Any], int]:
    """
    Modifie le pourcentage de trafic ML.
    Body:
        {
            "percentage": 50  // 0-100
        }
    Returns:
        JSON avec nouveau statut
    """
    try:
        data = request.get_json()
        if not data or "percentage" not in data:
            raise BadRequest("'percentage' field required in request body")

        percentage = data["percentage"]

        if not isinstance(percentage, int) or not 0 <= percentage <= 100:
            raise BadRequest("Percentage must be an integer between 0 and 100")

        FeatureFlags.set_ml_traffic_percentage(percentage)

        logger.warning(f"[FeatureFlagsAPI] ML percentage set to {percentage}% via API")

        return jsonify({
            "success": True,
            "message": f"Trafic ML configuré à {percentage}%",
            "status": get_feature_flags_status(),
        }), 200

    except BadRequest as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"[FeatureFlagsAPI] Error setting percentage: {e}")
        return jsonify({"error": str(e)}), 500


@feature_flags_bp.route("/reset-stats", methods=["POST"])
def reset_stats() -> tuple[dict[str, Any], int]:
    """
    Réinitialise les statistiques.
    Returns:
        JSON avec confirmation
    """
    try:
        FeatureFlags.reset_stats()

        logger.info("[FeatureFlagsAPI] Stats reset via API")

        return jsonify({
            "success": True,
            "message": "Statistiques réinitialisées",
            "status": get_feature_flags_status(),
        }), 200

    except Exception as e:
        logger.error(f"[FeatureFlagsAPI] Error resetting stats: {e}")
        return jsonify({"error": str(e)}), 500


@feature_flags_bp.route("/ml/health", methods=["GET"])
def ml_health() -> tuple[dict[str, Any], int]:
    """
    Vérifie la santé du système ML.
    Returns:
        JSON avec status de santé
    """
    try:
        status = get_feature_flags_status()
        stats = status["stats"]

        # Déterminer le status
        is_healthy = (
            status["config"]["ML_ENABLED"]
            and stats["ml_requests"] > 0
            and stats["ml_success_rate"] >= 0.95
        )

        health_status = "healthy" if is_healthy else "degraded"

        # Alertes
        alerts = []
        if stats["ml_success_rate"] < 0.95 and stats["ml_requests"] > 10:
            alerts.append(f"Taux de succès bas: {stats['ml_success_rate']:.1%}")

        if stats["ml_failures"] > 50:
            alerts.append(f"Trop d'erreurs: {stats['ml_failures']}")

        return jsonify({
            "status": health_status,
            "healthy": is_healthy,
            "ml_enabled": status["config"]["ML_ENABLED"],
            "ml_traffic": status["config"]["ML_TRAFFIC_PERCENTAGE"],
            "success_rate": stats["ml_success_rate"],
            "total_requests": stats["total_requests"],
            "alerts": alerts,
        }), 200 if is_healthy else 503

    except Exception as e:
        logger.error(f"[FeatureFlagsAPI] Error checking health: {e}")
        return jsonify({
            "status": "error",
            "healthy": False,
            "error": str(e),
        }), 500

