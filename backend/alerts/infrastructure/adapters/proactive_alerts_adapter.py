"""Adapter pour ProactiveAlertsService (compatibilité avec routes).

Encapsule ProactiveAlertsService pour permettre une migration progressive vers use-cases.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.proactive_alerts import ProactiveAlertsService


def create_proactive_alerts_service() -> ProactiveAlertsService:
    """Factory function pour créer ProactiveAlertsService.

    Returns:
        Instance de ProactiveAlertsService configurée pour la production.
    """
    return ProactiveAlertsService()


def check_delay_risk_via_service(
    booking: dict[str, Any],
    driver: dict[str, Any],
    current_time: datetime | None = None,
) -> dict[str, Any]:
    """Helper function pour analyser le risque de retard via service.

    Args:
        booking: Dictionnaire avec les données du booking.
        driver: Dictionnaire avec les données du driver.
        current_time: Temps actuel (par défaut: datetime.now(UTC)).

    Returns:
        Dictionnaire avec 'delay_probability', 'risk_level', 'explanation', etc.

    Raises:
        ValueError: Données invalides.
        RuntimeError: Erreur d'analyse.
    """
    service = create_proactive_alerts_service()
    if current_time is None:
        current_time = datetime.now(UTC)
    return service.check_delay_risk(
        booking=booking, driver=driver, current_time=current_time
    )


def get_rl_explanation_via_service(
    booking_id: str,
    driver_id: str,
    rl_decision: dict[str, Any],
) -> dict[str, Any]:
    """Helper function pour obtenir l'explication RL via service.

    Args:
        booking_id: ID du booking.
        driver_id: ID du driver.
        rl_decision: Dictionnaire avec les données de décision RL.

    Returns:
        Dictionnaire avec l'explication détaillée.

    Raises:
        ValueError: Données invalides.
        RuntimeError: Erreur de génération d'explication.
    """
    service = create_proactive_alerts_service()
    return service.get_explanation_for_decision(
        booking_id=booking_id, driver_id=driver_id, rl_decision=rl_decision
    )


def send_proactive_alert_via_service(
    analysis_result: dict[str, Any],
    company_id: str,
    force_send: bool = False,
) -> bool:
    """Helper function pour envoyer une alerte proactive via service.

    Args:
        analysis_result: Résultat de l'analyse de risque.
        company_id: ID de l'entreprise.
        force_send: Forcer l'envoi même si les seuils ne sont pas atteints.

    Returns:
        True si l'alerte a été envoyée, False sinon.

    Raises:
        ValueError: Données invalides.
        RuntimeError: Erreur d'envoi.
    """
    service = create_proactive_alerts_service()
    return service.send_proactive_alert(
        analysis_result=analysis_result, company_id=company_id, force_send=force_send
    )


def get_alert_statistics_via_service() -> dict[str, Any]:
    """Helper function pour obtenir les statistiques d'alertes via service.

    Returns:
        Dictionnaire avec les statistiques.

    Raises:
        RuntimeError: Erreur de récupération.
    """
    service = create_proactive_alerts_service()
    return service.get_alert_statistics()


def clear_alert_history_via_service(booking_id: str | None = None) -> None:
    """Helper function pour nettoyer l'historique d'alertes via service.

    Args:
        booking_id: ID du booking (optionnel, nettoie tout si None).

    Raises:
        RuntimeError: Erreur de nettoyage.
    """
    service = create_proactive_alerts_service()
    service.clear_alert_history(booking_id)
