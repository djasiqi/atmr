# Constantes pour éviter les valeurs magiques
import logging

from flask import Blueprint, jsonify, request  # pyright: ignore[reportMissingImports]
from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]

from ext import role_required
from models import UserRole
from security.ip_whitelist import ip_whitelist_required
from services.ml.monitoring import MLMonitoringService

HOURS_THRESHOLD = 24
HOURS_ONE = 1
DAYS_THRESHOLD = 30
DAYS_ONE = 1
LIMIT_THRESHOLD = 1000
LIMIT_ONE = 1
MAX_DAYS_LIMIT = 30

"""Routes API monitoring ML (F-05 : ADMIN + IP uniquement).

Endpoints:
    GET  /api/ml-monitoring/metrics        - Métriques temps réel
    GET  /api/ml-monitoring/daily          - Métriques par jour
    GET  /api/ml-monitoring/predictions    - Prédictions récentes
    GET  /api/ml-monitoring/anomalies      - Anomalies détectées
    GET  /api/ml-monitoring/summary        - Résumé complet
"""
# pyright: reportReturnType=false
# Flask jsonify() retourne Response, pas dict


logger = logging.getLogger(__name__)

# Créer le blueprint
ml_monitoring_bp = Blueprint("ml_monitoring", __name__, url_prefix="/api/ml-monitoring")


@ml_monitoring_bp.route("/metrics", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_metrics():
    """Récupère les métriques ML pour une période donnée."""
    try:
        hours = request.args.get("hours", 24, type=int)

        if hours < HOURS_ONE or hours > 24 * 30:  # Max 30 jours
            from shared.error_handlers import APIErrorHandler

            return APIErrorHandler.handle_validation_error(
                "hours must be between 1 and 720",
                field="hours",
                provided_value=hours,
                expected_format="1-720",
                logger_instance=logger,
            )

        metrics = MLMonitoringService.get_metrics(hours=hours)

        return jsonify(metrics), 200

    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)


@ml_monitoring_bp.route("/daily", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_daily_metrics():
    """Récupère les métriques par jour."""
    try:
        days = request.args.get("days", 7, type=int)

        if days < DAYS_ONE or days > MAX_DAYS_LIMIT:
            from shared.error_handlers import APIErrorHandler

            return APIErrorHandler.handle_validation_error(
                "days must be between 1 and 30",
                field="days",
                provided_value=days,
                expected_format="1-30",
                logger_instance=logger,
            )

        daily_metrics = MLMonitoringService.get_daily_metrics(days=days)

        return jsonify(
            {
                "days": days,
                "data": daily_metrics,
            }
        ), 200

    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)


@ml_monitoring_bp.route("/predictions", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_recent_predictions():
    """Récupère les prédictions récentes."""
    try:
        limit = request.args.get("limit", 100, type=int)

        if limit < LIMIT_ONE or limit > LIMIT_THRESHOLD:
            from shared.error_handlers import APIErrorHandler

            return APIErrorHandler.handle_validation_error(
                "limit must be between 1 and 1000",
                field="limit",
                provided_value=limit,
                expected_format="1-1000",
                logger_instance=logger,
            )

        predictions = MLMonitoringService.get_recent_predictions(limit=limit)

        return jsonify(
            {
                "limit": limit,
                "count": len(predictions),
                "predictions": predictions,
            }
        ), 200

    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)


@ml_monitoring_bp.route("/anomalies", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_anomalies():
    """Récupère les anomalies (prédictions très imprécises)."""
    try:
        threshold = request.args.get("threshold", 5.0, type=float)

        anomalies = MLMonitoringService.detect_anomalies(threshold_mae=threshold)

        return jsonify(
            {
                "threshold_mae": threshold,
                "count": len(anomalies),
                "anomalies": anomalies,
            }
        ), 200

    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)


@ml_monitoring_bp.route("/summary", methods=["GET"])
@jwt_required()
@role_required(UserRole.admin)
@ip_whitelist_required()
def get_summary():
    """Récupère le résumé complet du système ML."""
    try:
        summary = MLMonitoringService.get_summary()

        return jsonify(summary), 200

    except Exception as e:
        from shared.error_handlers import APIErrorHandler

        return APIErrorHandler.handle_exception(e, logger)
