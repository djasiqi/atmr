"""
Routes API pour le dashboard de monitoring ML.

Endpoints:
    GET  /api/ml-monitoring/metrics        - Métriques temps réel
    GET  /api/ml-monitoring/daily          - Métriques par jour
    GET  /api/ml-monitoring/predictions    - Prédictions récentes
    GET  /api/ml-monitoring/anomalies      - Anomalies détectées
    GET  /api/ml-monitoring/summary        - Résumé complet
"""
# ruff: noqa: ANN201
# pyright: reportReturnType=false
# Flask jsonify() retourne Response, pas dict
import logging
from typing import Any

from flask import Blueprint, jsonify, request

from services.ml_monitoring_service import MLMonitoringService

logger = logging.getLogger(__name__)

# Créer le blueprint
ml_monitoring_bp = Blueprint("ml_monitoring", __name__, url_prefix="/api/ml-monitoring")


@ml_monitoring_bp.route("/metrics", methods=["GET"])
def get_metrics():
    """
    Récupère les métriques ML pour une période donnée.
    
    Query params:
        hours: Nombre d'heures (défaut: 24)
    
    Returns:
        JSON avec MAE, R², accuracy_rate, etc.
    """
    try:
        hours = request.args.get("hours", 24, type=int)
        
        if hours < 1 or hours > 24 * 30:  # Max 30 jours
            return jsonify({"error": "hours must be between 1 and 720"}), 400
        
        metrics = MLMonitoringService.get_metrics(hours=hours)
        
        return jsonify(metrics), 200
        
    except Exception as e:
        logger.error(f"[MLMonitoringAPI] Error getting metrics: {e}")
        return jsonify({"error": str(e)}), 500


@ml_monitoring_bp.route("/daily", methods=["GET"])
def get_daily_metrics():
    """
    Récupère les métriques par jour.
    
    Query params:
        days: Nombre de jours (défaut: 7)
    
    Returns:
        JSON avec array de métriques par jour
    """
    try:
        days = request.args.get("days", 7, type=int)
        
        if days < 1 or days > 30:
            return jsonify({"error": "days must be between 1 and 30"}), 400
        
        daily_metrics = MLMonitoringService.get_daily_metrics(days=days)
        
        return jsonify({
            "days": days,
            "data": daily_metrics,
        }), 200
        
    except Exception as e:
        logger.error(f"[MLMonitoringAPI] Error getting daily metrics: {e}")
        return jsonify({"error": str(e)}), 500


@ml_monitoring_bp.route("/predictions", methods=["GET"])
def get_recent_predictions():
    """
    Récupère les prédictions récentes.
    
    Query params:
        limit: Nombre max (défaut: 100)
    
    Returns:
        JSON avec liste de prédictions
    """
    try:
        limit = request.args.get("limit", 100, type=int)
        
        if limit < 1 or limit > 1000:
            return jsonify({"error": "limit must be between 1 and 1000"}), 400
        
        predictions = MLMonitoringService.get_recent_predictions(limit=limit)
        
        return jsonify({
            "limit": limit,
            "count": len(predictions),
            "predictions": predictions,
        }), 200
        
    except Exception as e:
        logger.error(f"[MLMonitoringAPI] Error getting predictions: {e}")
        return jsonify({"error": str(e)}), 500


@ml_monitoring_bp.route("/anomalies", methods=["GET"])
def get_anomalies():
    """
    Récupère les anomalies (prédictions très imprécises).
    
    Query params:
        threshold: Seuil MAE pour anomalie (défaut: 5.0 min)
    
    Returns:
        JSON avec liste d'anomalies
    """
    try:
        threshold = request.args.get("threshold", 5.0, type=float)
        
        anomalies = MLMonitoringService.detect_anomalies(threshold_mae=threshold)
        
        return jsonify({
            "threshold_mae": threshold,
            "count": len(anomalies),
            "anomalies": anomalies,
        }), 200
        
    except Exception as e:
        logger.error(f"[MLMonitoringAPI] Error getting anomalies: {e}")
        return jsonify({"error": str(e)}), 500


@ml_monitoring_bp.route("/summary", methods=["GET"])
def get_summary():
    """
    Récupère le résumé complet du système ML.
    
    Returns:
        JSON avec toutes les métriques importantes
    """
    try:
        summary = MLMonitoringService.get_summary()
        
        return jsonify(summary), 200
        
    except Exception as e:
        logger.error(f"[MLMonitoringAPI] Error getting summary: {e}")
        return jsonify({"error": str(e)}), 500

