# backend/services/monitoring/notification_metrics.py
"""Métriques Prometheus pour le système de notifications.

Phase 2 - Enrichissement
Permet de suivre l'efficacité et la performance des notifications.
"""

from typing import Any

from prometheus_client import Counter, Histogram  # type: ignore[import-untyped]

# ✅ Compteurs de notifications envoyées
notifications_sent_total = Counter(
    "notifications_sent_total",
    "Nombre total de notifications envoyées",
    ["notification_type", "channel", "status"],
)

# ✅ Compteurs d'actions sur notifications
notification_actions_total = Counter(
    "notification_actions_total",
    "Actions effectuées sur les notifications",
    ["notification_type", "action_type"],
)

# ✅ Compteurs d'ouvertures
notifications_opened_total = Counter(
    "notifications_opened_total",
    "Notifications ouvertes par les utilisateurs",
    ["notification_type"],
)

# ✅ Compteurs d'échecs
notifications_failed_total = Counter(
    "notifications_failed_total",
    "Notifications en échec",
    ["notification_type", "error_reason"],
)

# ✅ Compteurs mode nuit
notifications_skipped_night_total = Counter(
    "notifications_skipped_night_total",
    "Notifications bloquées par le mode nuit",
    ["notification_type", "reason"],
)

# ✅ Histogramme temps de délivrance
notification_delivery_duration_seconds = Histogram(
    "notification_delivery_duration_seconds",
    "Temps de délivrance des notifications",
    ["notification_type"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

# ✅ Histogramme temps de réponse (action)
notification_action_response_duration_seconds = Histogram(
    "notification_action_response_duration_seconds",
    "Temps entre notification et action utilisateur",
    ["notification_type", "action_type"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0, 1800.0),
)

# ✅ Métriques notifications silencieuses (Phase 2.6)
silent_notifications_sent_total = Counter(
    "silent_notifications_sent_total",
    "Notifications silencieuses envoyées pour sync données",
    ["sync_type", "status"],
)

silent_sync_duration_seconds = Histogram(
    "silent_sync_duration_seconds",
    "Durée de synchronisation données silencieuses",
    ["sync_type"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)


# ========================================
# Helpers pour incrémenter les métriques
# ========================================


def track_notification_sent(
    notification_type: str, channel: str, status: str = "success"
) -> None:
    """Enregistre une notification envoyée.

    Args:
        notification_type: Type de notification (booking, message, etc.)
        channel: Canal utilisé (missions, critical, etc.)
        status: Statut (success, failed)
    """
    notifications_sent_total.labels(
        notification_type=notification_type, channel=channel, status=status
    ).inc()


def track_notification_action(notification_type: str, action_type: str) -> None:
    """Enregistre une action sur notification.

    Args:
        notification_type: Type de notification
        action_type: Type d'action (accept, reject, view, etc.)
    """
    notification_actions_total.labels(
        notification_type=notification_type, action_type=action_type
    ).inc()


def track_notification_opened(notification_type: str) -> None:
    """Enregistre l'ouverture d'une notification.

    Args:
        notification_type: Type de notification
    """
    notifications_opened_total.labels(notification_type=notification_type).inc()


def track_notification_failed(notification_type: str, error_reason: str) -> None:
    """Enregistre un échec de notification.

    Args:
        notification_type: Type de notification
        error_reason: Raison de l'échec (network, token_invalid, etc.)
    """
    notifications_failed_total.labels(
        notification_type=notification_type, error_reason=error_reason
    ).inc()


def track_notification_skipped_night(
    notification_type: str, reason: str = "night_mode"
) -> None:
    """Enregistre une notification bloquée par le mode nuit.

    Args:
        notification_type: Type de notification
        reason: Raison du blocage (driver_off_duty, night_mode, etc.)
    """
    notifications_skipped_night_total.labels(
        notification_type=notification_type, reason=reason
    ).inc()


def track_notification_delivery_duration(
    notification_type: str, duration_seconds: float
) -> None:
    """Enregistre le temps de délivrance d'une notification.

    Args:
        notification_type: Type de notification
        duration_seconds: Durée en secondes
    """
    notification_delivery_duration_seconds.labels(
        notification_type=notification_type
    ).observe(duration_seconds)


def track_notification_action_response_duration(
    notification_type: str, action_type: str, duration_seconds: float
) -> None:
    """Enregistre le temps de réponse à une notification.

    Args:
        notification_type: Type de notification
        action_type: Type d'action
        duration_seconds: Durée en secondes entre notification et action
    """
    notification_action_response_duration_seconds.labels(
        notification_type=notification_type, action_type=action_type
    ).observe(duration_seconds)


def track_silent_notification_sent(sync_type: str, status: str = "success") -> None:
    """Enregistre une notification silencieuse envoyée.

    Args:
        sync_type: Type de sync (missions, profile, maps, etc.)
        status: Statut (success, failed)
    """
    silent_notifications_sent_total.labels(sync_type=sync_type, status=status).inc()


def track_silent_sync_duration(sync_type: str, duration_seconds: float) -> None:
    """Enregistre la durée d'une synchronisation silencieuse.

    Args:
        sync_type: Type de sync
        duration_seconds: Durée en secondes
    """
    silent_sync_duration_seconds.labels(sync_type=sync_type).observe(duration_seconds)


# ========================================
# Fonction de calcul des KPIs
# ========================================


def get_notification_metrics_summary() -> dict[str, Any]:
    """Retourne un résumé des métriques de notifications.

    Utile pour debugging et dashboards.

    Returns:
        Dict contenant les métriques clés
    """
    # Note: Prometheus ne permet pas de récupérer facilement les valeurs
    # Cette fonction est plutôt pour documentation
    # En pratique, utiliser les dashboards Grafana

    return {
        "info": "Utilisez Grafana pour visualiser les métriques Prometheus",
        "metrics_available": [
            "notifications_sent_total",
            "notification_actions_total",
            "notifications_opened_total",
            "notifications_failed_total",
            "notifications_skipped_night_total",
            "notification_delivery_duration_seconds",
            "notification_action_response_duration_seconds",
            "silent_notifications_sent_total",
            "silent_sync_duration_seconds",
        ],
        "example_queries": {
            "open_rate": 'notifications_opened_total / notifications_sent_total{status="success"}',
            "action_rate": "notification_actions_total / notifications_opened_total",
            "night_skip_rate": "notifications_skipped_night_total / (notifications_sent_total + notifications_skipped_night_total)",
            "avg_delivery_time": "rate(notification_delivery_duration_seconds_sum[5m]) / rate(notification_delivery_duration_seconds_count[5m])",
            "avg_response_time": "rate(notification_action_response_duration_seconds_sum[5m]) / rate(notification_action_response_duration_seconds_count[5m])",
            "silent_sync_success_rate": 'silent_notifications_sent_total{status="success"} / silent_notifications_sent_total',
            "avg_sync_duration": "rate(silent_sync_duration_seconds_sum[5m]) / rate(silent_sync_duration_seconds_count[5m])",
        },
    }
