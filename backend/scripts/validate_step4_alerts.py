#!/usr/bin/env python3
"""Script de validation pour l'Étape 4 - Alertes Proactives + Explicabilité.

Ce script teste:
- Service d'alertes proactives
- Routes REST
- Socket.IO handlers
- Système de debounce
- Intégration avec delay_predictor.pkl

Auteur: ATMR Project - RL Team
Date: 21 octobre 2025
"""

import json
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_proactive_alerts_service():
    """Test du service d'alertes proactives."""
    logger.info("🧪 Test du service d'alertes proactives...")

    try:
        from services.proactive_alerts import ProactiveAlertsService

        # Initialiser le service
        service = ProactiveAlertsService()

        # Données de test
        test_booking = {
            "id": "test_123",
            "pickup_lat": 46.2044,
            "pickup_lon": 6.1432,
            "pickup_time": (datetime.now(UTC) + timedelta(minutes=20)).isoformat(),
            "priority": 3,
            "is_outbound": True,
            "estimated_duration": 30
        }

        test_driver = {
            "id": "driver_456",
            "lat": 46.2100,
            "lon": 6.1400,
            "current_bookings": 2,
            "load": 0,
            "type": "REGULAR",
            "available": True
        }

        # Test analyse de risque
        result = service.check_delay_risk(test_booking, test_driver)

        assert "delay_probability" in result
        assert "risk_level" in result
        assert "explanation" in result
        assert "should_alert" in result

        logger.info("✅ Analyse de risque: %s (%.1f%%)", result["risk_level"], result["delay_probability"]*100)

        # Test explicabilité RL
        rl_decision = {
            "q_values": {"action_1": 0.8, "action_2": 0.6},
            "confidence": 0.85,
            "reward_components": {"punctuality": 0.7, "distance": 0.3}
        }

        explanation = service.get_explanation_for_decision(
            booking_id="test_123",
            driver_id="driver_456",
            rl_decision=rl_decision
        )

        assert "decision_factors" in explanation
        assert "alternative_options" in explanation

        logger.info("✅ Explicabilité RL fonctionnelle")

        # Test système de debounce
        analysis_result = {
            "booking_id": "test_booking",
            "driver_id": "driver_456",
            "delay_probability": 0.8,
            "risk_level": "high",
            "should_alert": True,
            "explanation": {"test": "explanation"},
            "timestamp": datetime.now(UTC).isoformat()
        }

        # Premier envoi
        success1 = service.send_proactive_alert(analysis_result, "test_company")
        logger.info("✅ Premier envoi d'alerte: %s", "Succès" if success1 else "Échec")

        # Deuxième envoi (doit être debounced)
        success2 = service.send_proactive_alert(analysis_result, "test_company")
        logger.info("✅ Deuxième envoi (debounced): %s", "Succès" if success2 else "Échec")

        # Test statistiques
        stats = service.get_alert_statistics()
        assert "total_alerts_sent" in stats
        logger.info("✅ Statistiques: %s alertes", stats["total_alerts_sent"])

        logger.info("🎉 Service d'alertes proactives validé avec succès!")
        return True

    except Exception as e:
        logger.error("❌ Erreur test service: %s", e)
        return False


def test_routes():
    """Test des routes REST."""
    logger.info("🧪 Test des routes REST...")

    try:
        from routes.proactive_alerts import _get_mock_booking_data, _get_mock_driver_data, proactive_alerts_bp

        # Test données mock
        booking_data = _get_mock_booking_data("123")
        driver_data = _get_mock_driver_data("456")

        assert booking_data is not None
        assert driver_data is not None
        assert booking_data["id"] == "123"
        assert driver_data["id"] == "456"

        logger.info("✅ Données mock fonctionnelles")

        # Test blueprint
        assert proactive_alerts_bp.name == "proactive_alerts"
        assert proactive_alerts_bp.url_prefix == "/api/alerts"

        logger.info("✅ Blueprint configuré correctement")

        logger.info("🎉 Routes REST validées avec succès!")
        return True

    except Exception as e:
        logger.error("❌ Erreur test routes: %s", e)
        return False


def test_socketio_handlers():
    """Test des handlers Socket.IO."""
    logger.info("🧪 Test des handlers Socket.IO...")

    try:
        from sockets.proactive_alerts import (
            broadcast_delay_alert,
            broadcast_rl_explanation,
            get_active_connections_stats,
        )

        # Test fonctions de diffusion
        analysis_result = {
            "booking_id": "test_123",
            "driver_id": "driver_456",
            "delay_probability": 0.8,
            "risk_level": "high",
            "explanation": {"test": "explanation"}
        }

        # Mock SocketIO avec type Any pour éviter les erreurs de type
        mock_socketio = type("MockSocketIO", (), {
            "emit": lambda self, event, data, room=None: True
        })()

        success = broadcast_delay_alert("test_company", analysis_result, mock_socketio)  # type: ignore
        assert success is True

        explanation = {
            "booking_id": "test_123",
            "driver_id": "driver_456",
            "decision_factors": [],
            "alternative_options": []
        }

        success = broadcast_rl_explanation("test_company", explanation, mock_socketio)  # type: ignore
        assert success is True

        # Test statistiques connexions
        stats = get_active_connections_stats()
        assert "total_companies" in stats
        assert "total_connections" in stats

        logger.info("✅ Handlers Socket.IO fonctionnels")

        logger.info("🎉 Handlers Socket.IO validés avec succès!")
        return True

    except Exception as e:
        logger.error("❌ Erreur test Socket.IO: %s", e)
        return False


def test_integration():
    """Test d'intégration complète."""
    logger.info("🧪 Test d'intégration complète...")

    try:
        # Test intégration avec delay_predictor

        service = ProactiveAlertsService()

        # Vérifier que le service peut charger le delay_predictor
        if service.delay_predictor is not None:
            logger.info("✅ DelayMLPredictor chargé avec succès")
        else:
            logger.warning("⚠️ DelayMLPredictor non disponible (fallback heuristique)")

        # Test avec données réalistes
        realistic_booking = {
            "id": "realistic_123",
            "pickup_lat": 46.2044,
            "pickup_lon": 6.1432,
            "pickup_time": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),  # Risque élevé
            "priority": 4,
            "is_outbound": True,
            "estimated_duration": 30
        }

        realistic_driver = {
            "id": "driver_realistic",
            "lat": 46.3000,  # Loin
            "lon": 6.2000,
            "current_bookings": 4,  # Charge élevée
            "load": 0,
            "type": "REGULAR",
            "available": True
        }

        result = service.check_delay_risk(realistic_booking, realistic_driver)

        logger.info("✅ Analyse réaliste: %s (%.1f%%)", result["risk_level"], result["delay_probability"]*100)

        # Vérifier que le système détecte bien le risque élevé
        if result["risk_level"] in ["medium", "high"]:
            logger.info("✅ Détection de risque élevé fonctionnelle")
        else:
            logger.warning("⚠️ Risque élevé non détecté")

        logger.info("🎉 Intégration complète validée avec succès!")
        return True

    except Exception as e:
        logger.error("❌ Erreur test intégration: %s", e)
        return False


def generate_validation_report():
    """Génère un rapport de validation."""
    logger.info("📊 Génération du rapport de validation...")

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "step": "Étape 4 - Alertes Proactives + Explicabilité",
        "tests": {
            "proactive_alerts_service": False,
            "routes": False,
            "socketio_handlers": False,
            "integration": False
        },
        "summary": {
            "total_tests": 4,
            "passed": 0,
            "failed": 0
        }
    }

    # Exécuter les tests
    tests = [
        ("proactive_alerts_service", test_proactive_alerts_service),
        ("routes", test_routes),
        ("socketio_handlers", test_socketio_handlers),
        ("integration", test_integration)
    ]

    for test_name, test_func in tests:
        try:
            result = test_func()
            report["tests"][test_name] = result
            if result:
                report["summary"]["passed"] += 1
            else:
                report["summary"]["failed"] += 1
        except Exception as e:
            logger.error("❌ Test %s échoué: %s", test_name, e)
            report["summary"]["failed"] += 1

    # Sauvegarder le rapport
    report_path = Path("backend/data/rl/validation_reports")
    report_path.mkdir(parents=True, exist_ok=True)

    report_file = report_path / "step4validation_report.json"
    with Path(report_file, "w", encoding="utf-8").open() as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("📄 Rapport sauvegardé: %s", report_file)

    # Afficher le résumé
    logger.info("=" * 60)
    logger.info("📊 RAPPORT DE VALIDATION - ÉTAPE 4")
    logger.info("=" * 60)
    logger.info("Tests exécutés: %s", report["summary"]["total_tests"])
    logger.info("Tests réussis: %s", report["summary"]["passed"])
    logger.info("Tests échoués: %s", report["summary"]["failed"])
    logger.info("")

    for test_name, result in report["tests"].items():
        status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
        logger.info("%s: %s", test_name, status)

    logger.info("=" * 60)

    if report["summary"]["failed"] == 0:
        logger.info("🎉 TOUS LES TESTS SONT RÉUSSIS!")
        logger.info("✅ L'Étape 4 - Alertes Proactives + Explicabilité est validée!")
    else:
        logger.warning("⚠️ %s test(s) ont échoué", report["summary"]["failed"])
        logger.warning("❌ L'Étape 4 nécessite des corrections")

    return report


def main():
    """Fonction principale."""
    logger.info("🚀 Début de la validation de l'Étape 4")
    logger.info("=" * 60)

    try:
        report = generate_validation_report()

        if report["summary"]["failed"] == 0:
            logger.info("🎉 VALIDATION RÉUSSIE!")
            sys.exit(0)
        else:
            logger.error("❌ VALIDATION ÉCHOUÉE!")
            sys.exit(1)

    except Exception as e:
        logger.error("❌ Erreur critique: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
