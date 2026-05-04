"""Handler générique pour collecter des métriques sur tous les événements.

Migration progressive vers Clean Architecture:
- Centralise la collecte de métriques pour tous les événements
- Permet d'ajouter facilement de nouveaux types de métriques
- Découplé des systèmes de métriques spécifiques (Prometheus, StatsD, etc.)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Compteurs en mémoire (pour développement/test)
# En production, utiliser Prometheus, StatsD, ou DB
_event_counters: dict[str, int] = {}


def handle_event_metrics(event: dict[str, Any]) -> None:
    """Collecte des métriques pour tous les événements.

    Actions:
    - Incrémente compteur par type d'événement
    - Enregistre métriques spécifiques selon le type d'événement
    - Log les métriques pour observabilité

    Args:
        event: Dictionnaire d'événement avec au minimum 'event_type'
    """
    event_type = str(event.get("event_type") or "")
    if not event_type:
        logger.warning("[EventBus] Metrics: event_type manquant, ignore")
        return

    try:
        # Incrémenter compteur global par type d'événement
        _event_counters[event_type] = _event_counters.get(event_type, 0) + 1

        # Métriques spécifiques par type d'événement
        if event_type == "BookingCreatedEvent":
            _handle_booking_created_metrics(event)
        elif event_type == "BookingAssignedEvent":
            _handle_booking_assigned_metrics(event)
        elif event_type == "DriverLocationUpdatedEvent":
            _handle_driver_location_metrics(event)
        elif event_type == "DispatchRunCompletedEvent":
            _handle_dispatch_completed_metrics(event)

        logger.debug(
            "[EventBus] Metrics collected for %s (total: %d)",
            event_type,
            _event_counters.get(event_type, 0),
        )
    except (KeyError, TypeError, AttributeError) as e:
        # Erreurs de validation attendues : clés manquantes, types incorrects
        logger.warning(
            "[EventBus] Metrics collection failed for %s (validation error: %s): %s",
            event_type,
            type(e).__name__,
            e,
        )
    except Exception:
        # Handler "safe" : ne pas faire échouer le système si métriques échouent
        logger.exception("[EventBus] Metrics collection failed for %s", event_type)


def _handle_booking_created_metrics(event: dict[str, Any]) -> None:
    """Métriques spécifiques pour BookingCreatedEvent."""
    booking_id = event.get("booking_id")
    company_id = event.get("company_id")

    # Ici on pourrait :
    # - Enregistrer dans DispatchMetrics ou système d'analytics
    # - Envoyer à Prometheus/StatsD
    # - Calculer temps de création, source, etc.

    logger.debug(
        "[EventBus] BookingCreated metrics: booking_id=%s company_id=%s",
        booking_id,
        company_id,
    )


def _handle_booking_assigned_metrics(event: dict[str, Any]) -> None:
    """Métriques spécifiques pour BookingAssignedEvent."""
    booking_id = event.get("booking_id")
    driver_id = event.get("driver_id")

    logger.debug(
        "[EventBus] BookingAssigned metrics: booking_id=%s driver_id=%s",
        booking_id,
        driver_id,
    )


def _handle_driver_location_metrics(event: dict[str, Any]) -> None:
    """Métriques spécifiques pour DriverLocationUpdatedEvent."""
    driver_id = event.get("driver_id")
    company_id = event.get("company_id")

    # Ici on pourrait :
    # - Calculer vitesse moyenne (nécessiterait enrichir l'événement avec speed)
    # - Enregistrer distance parcourue
    # - Détecter arrêts anormaux
    # Note: Les détails de position sont gérés par LocationService

    logger.debug(
        "[EventBus] DriverLocation metrics: driver_id=%s company_id=%s",
        driver_id,
        company_id,
    )


def _handle_dispatch_completed_metrics(event: dict[str, Any]) -> None:
    """Métriques spécifiques pour DispatchRunCompletedEvent."""
    company_id = event.get("company_id")
    assignments_count = event.get("assignments_count", 0)

    logger.debug(
        "[EventBus] DispatchCompleted metrics: company_id=%s assignments=%d",
        company_id,
        assignments_count,
    )


def get_event_counters() -> dict[str, int]:
    """Retourne les compteurs d'événements (pour tests/monitoring).

    Returns:
        Dictionnaire {event_type: count}
    """
    return _event_counters.copy()


def reset_counters() -> None:
    """Réinitialise les compteurs (pour tests uniquement)."""
    global _event_counters
    _event_counters = {}
