#!/usr/bin/env python3
"""Routes REST pour les alertes proactives et l'explicabilité RL.

Ce module expose les endpoints pour:
- Analyse de risque de retard
- Explicabilité des décisions RL
- Gestion des alertes proactives
- Statistiques et monitoring

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any, Dict

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required

from services.proactive_alerts import ProactiveAlertsService

logger = logging.getLogger(__name__)

# Créer le blueprint
proactive_alerts_bp = Blueprint("proactive_alerts", __name__, url_prefix="/api/alerts")

# Instance globale du service
alerts_service = ProactiveAlertsService()


@proactive_alerts_bp.route("/delay-risk", methods=["GET"])
@jwt_required()
def check_delay_risk():
    """Endpoint pour analyser le risque de retard d'une assignation.
    Query Parameters:
    - booking_id: ID du booking
    - driver_id: ID du chauffeur
    - company_id: ID de l'entreprise
    Returns:
        Analyse de risque avec probabilité et explication.
    """
    try:
        # Récupérer les paramètres
        booking_id = request.args.get("booking_id")
        driver_id = request.args.get("driver_id")
        company_id = request.args.get("company_id")

        if not all([booking_id, driver_id, company_id]):
            return jsonify(
                {"error": "Paramètres manquants", "required": ["booking_id", "driver_id", "company_id"]}
            ), 400

        # TODO: Récupérer les données depuis la base de données
        # Pour l'instant, simulation avec données mock
        if booking_id and driver_id:
            booking_data = _get_mock_booking_data(booking_id)
            driver_data = _get_mock_driver_data(driver_id)
        else:
            booking_data = None
            driver_data = None

        if not booking_data or not driver_data:
            return jsonify({"error": "Booking ou chauffeur non trouvé"}), 404

        # Analyser le risque
        analysis_result = alerts_service.check_delay_risk(
            booking=booking_data, driver=driver_data, current_time=datetime.now(UTC)
        )

        logger.info(
            "[ProactiveAlerts] Analyse risque - Booking %s, Driver %s: %.1f%% (%s)",
            booking_id,
            driver_id,
            analysis_result["delay_probability"] * 100,
            analysis_result["risk_level"],
        )

        return jsonify({"success": True, "analysis": analysis_result, "timestamp": datetime.now(UTC).isoformat()})

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur endpoint delay-risk: %s", e)
        return jsonify({"error": "Erreur interne du serveur", "details": str(e)}), 500


@proactive_alerts_bp.route("/delay-risk", methods=["POST"])
@jwt_required()
def analyze_multiple_delay_risks():
    """Endpoint pour analyser plusieurs risques de retard en batch.
    Body:
        {
            "assignments": [
                {"booking_id": "123", "driver_id": "456"},
                {"booking_id": "124", "driver_id": "457"}
            ],
            "company_id": "company_123"
        }.

    Returns:
        Liste des analyses de risque

    """
    try:
        data = request.get_json()

        if not data or "assignments" not in data:
            return jsonify({"error": "Body manquant ou invalide", "required": ["assignments", "company_id"]}), 400

        assignments = data["assignments"]
        company_id = data.get("company_id")

        if not company_id:
            return jsonify({"error": "company_id requis"}), 400

        if not isinstance(assignments, list) or len(assignments) == 0:
            return jsonify({"error": "assignments doit être une liste non vide"}), 400

        # Analyser chaque assignation
        results = []
        for assignment in assignments:
            booking_id = assignment.get("booking_id")
            driver_id = assignment.get("driver_id")

            if not booking_id or not driver_id:
                results.append(
                    {"booking_id": booking_id, "driver_id": driver_id, "error": "booking_id et driver_id requis"}
                )
                continue

            # Récupérer données
            booking_data = _get_mock_booking_data(booking_id)
            driver_data = _get_mock_driver_data(driver_id)

            if not booking_data or not driver_data:
                results.append({"booking_id": booking_id, "driver_id": driver_id, "error": "Données non trouvées"})
                continue

            # Analyser
            analysis_result = alerts_service.check_delay_risk(
                booking=booking_data, driver=driver_data, current_time=datetime.now(UTC)
            )

            results.append({"booking_id": booking_id, "driver_id": driver_id, "analysis": analysis_result})

        logger.info("[ProactiveAlerts] Analyse batch - %d assignations analysées", len(results))

        return jsonify(
            {
                "success": True,
                "results": results,
                "total_analyzed": len(results),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur endpoint batch delay-risk: %s", e)
        return jsonify({"error": "Erreur interne du serveur", "details": str(e)}), 500


@proactive_alerts_bp.route("/explain-decision", methods=["POST"])
@jwt_required()
def explain_rl_decision():
    """Endpoint pour expliquer une décision RL.
    Body:
        {
            "booking_id": "123",
            "driver_id": "456",
            "rl_decision": {
                "q_values": {"action_1": 0.8, "action_2": 0.6},
                "confidence": 0.85,
                "reward_components": {"punctuality": 0.7, "distance": 0.3}
            }
        }.

    Returns:
        Explication détaillée de la décision RL

    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Body requis"}), 400

        booking_id = data.get("booking_id")
        driver_id = data.get("driver_id")
        rl_decision = data.get("rl_decision", {})

        if not all([booking_id, driver_id]):
            return jsonify({"error": "booking_id et driver_id requis"}), 400

        # Générer l'explication
        explanation = alerts_service.get_explanation_for_decision(
            booking_id=booking_id, driver_id=driver_id, rl_decision=rl_decision
        )

        logger.info("[ProactiveAlerts] Explication générée - Booking %s, Driver %s", booking_id, driver_id)

        return jsonify({"success": True, "explanation": explanation, "timestamp": datetime.now(UTC).isoformat()})

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur endpoint explain-decision: %s", e)
        return jsonify({"error": "Erreur interne du serveur", "details": str(e)}), 500


@proactive_alerts_bp.route("/send-alert", methods=["POST"])
@jwt_required()
def send_proactive_alert():
    """Endpoint pour envoyer une alerte proactive.
    Body:
        {
            "booking_id": "123",
            "driver_id": "456",
            "company_id": "company_123",
            "force_send": false
        }.

    Returns:
        Statut de l'envoi de l'alerte

    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Body requis"}), 400

        booking_id = data.get("booking_id")
        driver_id = data.get("driver_id")
        company_id = data.get("company_id")
        force_send = data.get("force_send", False)

        if not all([booking_id, driver_id, company_id]):
            return jsonify({"error": "booking_id, driver_id et company_id requis"}), 400

        # Récupérer données
        booking_data = _get_mock_booking_data(booking_id)
        driver_data = _get_mock_driver_data(driver_id)

        if not booking_data or not driver_data:
            return jsonify({"error": "Booking ou chauffeur non trouvé"}), 404

        # Analyser le risque
        analysis_result = alerts_service.check_delay_risk(
            booking=booking_data, driver=driver_data, current_time=datetime.now(UTC)
        )

        # Envoyer l'alerte
        alert_sent = alerts_service.send_proactive_alert(
            analysis_result=analysis_result, company_id=company_id, force_send=force_send
        )

        logger.info(
            "[ProactiveAlerts] Alerte %s - Booking %s, Company %s",
            "envoyée" if alert_sent else "non envoyée",
            booking_id,
            company_id,
        )

        return jsonify(
            {
                "success": True,
                "alert_sent": alert_sent,
                "analysis": analysis_result,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur endpoint send-alert: %s", e)
        return jsonify({"error": "Erreur interne du serveur", "details": str(e)}), 500


@proactive_alerts_bp.route("/statistics", methods=["GET"])
@jwt_required()
def get_alert_statistics():
    """Endpoint pour récupérer les statistiques des alertes.

    Returns:
        Statistiques des alertes proactives

    """
    try:
        stats = alerts_service.get_alert_statistics()

        logger.info("[ProactiveAlerts] Statistiques récupérées")

        return jsonify({"success": True, "statistics": stats, "timestamp": datetime.now(UTC).isoformat()})

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur endpoint statistics: %s", e)
        return jsonify({"error": "Erreur interne du serveur", "details": str(e)}), 500


@proactive_alerts_bp.route("/clear-history", methods=["POST"])
@jwt_required()
def clear_alert_history():
    """Endpoint pour nettoyer l'historique des alertes.
    Body (optionnel):
        {
            "booking_id": "123"  # Si spécifié, nettoie seulement ce booking
        }.

    Returns:
        Confirmation du nettoyage

    """
    try:
        data = request.get_json() or {}

        # ✅ 2.4: Validation Marshmallow avec erreurs 400 détaillées
        from marshmallow import ValidationError

        from schemas.alert_schemas import ClearAlertHistorySchema
        from schemas.validation_utils import handle_validation_error, validate_request

        try:
            validated_data = validate_request(ClearAlertHistorySchema(), data, strict=False)
        except ValidationError as e:
            return handle_validation_error(e)

        booking_id = validated_data.get("booking_id")

        # Nettoyer l'historique
        alerts_service.clear_alert_history(booking_id)

        logger.info(
            "[ProactiveAlerts] Historique nettoyé%s", f" pour booking {booking_id}" if booking_id else " (complet)"
        )

        return jsonify(
            {
                "success": True,
                "message": f"Historique nettoyé{' pour booking ' + booking_id if booking_id else ''}",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur endpoint clear-history: %s", e)
        return jsonify({"error": "Erreur interne du serveur", "details": str(e)}), 500


@proactive_alerts_bp.route("/health", methods=["GET"])
def health_check():
    """Endpoint de santé pour le service d'alertes proactives.

    Returns:
        Statut de santé du service

    """
    try:
        # Vérifier les composants
        delay_predictor_loaded = alerts_service.delay_predictor is not None
        notification_service_available = alerts_service.notification_service is not None

        health_status = {
            "service": "proactive_alerts",
            "status": "healthy",
            "components": {
                "delay_predictor": "loaded" if delay_predictor_loaded else "not_loaded",
                "notification_service": "available" if notification_service_available else "unavailable",
                "alerts_service": "running",
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

        # Déterminer le statut global
        if not delay_predictor_loaded:
            health_status["status"] = "degraded"
            health_status["warnings"] = ["delay_predictor not loaded - using heuristic fallback"]

        return jsonify(health_status)

    except Exception as e:
        logger.error("[ProactiveAlerts] Erreur health check: %s", e)
        return jsonify(
            {
                "service": "proactive_alerts",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        ), 500


# Fonctions utilitaires pour données mock
def _get_mock_booking_data(booking_id: str) -> Dict[str, Any] | None:
    """Récupère des données mock pour un booking."""
    # TODO: Remplacer par vraie requête DB
    mock_bookings = {
        "123": {
            "id": "123",
            "pickup_lat": 46.2044,
            "pickup_lon": 6.1432,
            "pickup_time": (datetime.now(UTC).replace(microsecond=0) + timedelta(minutes=25)).isoformat(),
            "priority": 3,
            "is_outbound": True,
            "estimated_duration": 30,
        },
        "124": {
            "id": "124",
            "pickup_lat": 46.2200,
            "pickup_lon": 6.1500,
            "pickup_time": (datetime.now(UTC).replace(microsecond=0) + timedelta(minutes=45)).isoformat(),
            "priority": 4,
            "is_outbound": False,
            "estimated_duration": 25,
        },
    }

    return mock_bookings.get(booking_id)


def _get_mock_driver_data(driver_id: str) -> Dict[str, Any] | None:
    """Récupère des données mock pour un chauffeur."""
    # TODO: Remplacer par vraie requête DB
    mock_drivers = {
        "456": {
            "id": "456",
            "lat": 46.2100,
            "lon": 6.1400,
            "current_bookings": 2,
            "load": 0,
            "type": "REGULAR",
            "available": True,
        },
        "457": {
            "id": "457",
            "lat": 46.2000,
            "lon": 6.1300,
            "current_bookings": 1,
            "load": 0,
            "type": "EMERGENCY",
            "available": True,
        },
    }

    return mock_drivers.get(driver_id)


# Fonction pour enregistrer le blueprint dans l'app Flask
def register_proactive_alerts_routes(app):
    """Enregistre les routes d'alertes proactives dans l'application Flask."""
    app.register_blueprint(proactive_alerts_bp)
    logger.info("[ProactiveAlerts] Routes enregistrées: /api/alerts/*")
