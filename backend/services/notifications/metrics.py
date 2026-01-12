# backend/services/notifications/metrics.py
"""Métriques Prometheus pour le système de notifications.

Expose toutes les métriques pour Grafana dashboards.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (  # pyright: ignore[reportMissingImports]
        Counter,
        Gauge,
        Histogram,
    )

    # Compteurs
    NOTIFICATIONS_SENT = Counter(
        "notifications_sent_total",
        "Total notifications envoyées",
        ["channel", "notification_type", "region"],
    )

    NOTIFICATIONS_FAILED = Counter(
        "notifications_failed_total",
        "Total notifications échouées",
        ["channel", "notification_type", "reason", "region"],
    )

    NOTIFICATIONS_DEDUPLICATED = Counter(
        "notifications_deduplicated_total",
        "Notifications déduplicatées",
        ["notification_type", "region"],
    )

    PUSH_TOKENS_INVALIDATED = Counter(
        "push_tokens_invalidated_total",
        "Tokens push invalidés",
        ["reason", "region"],
    )

    CIRCUIT_BREAKER_OPENED = Counter(
        "circuit_breaker_opened_total",
        "Nombre de fois où le circuit breaker s'est ouvert",
        ["service", "region"],
    )

    REGION_FAILOVERS = Counter(
        "region_failovers_total",
        "Nombre de failovers entre régions",
        ["from_region", "to_region"],
    )

    # Gauges
    CIRCUIT_BREAKER_STATE = Gauge(
        "circuit_breaker_state",
        "État du circuit breaker (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
        ["service", "region"],
    )

    ACTIVE_REGION = Gauge(
        "active_region",
        "Région active (1=eu-west-1, 2=eu-central-1)",
        ["region"],
    )

    REDIS_QUEUE_LENGTH = Gauge(
        "redis_queue_length",
        "Longueur de la queue Redis",
        ["queue", "region"],
    )

    KAFKA_CONSUMER_LAG = Gauge(
        "kafka_consumer_lag",
        "Lag du consumer Kafka",
        ["topic", "partition", "consumer_group"],
    )

    DLQ_MESSAGES = Gauge(
        "dlq_messages_total",
        "Nombre de messages en Dead Letter Queue",
        ["source", "region"],
    )

    # Histogrammes (latence)
    PUSH_NOTIFICATION_DURATION = Histogram(
        "push_notification_duration_seconds",
        "Durée d'envoi d'une notification push",
        ["region"],
        buckets=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
    )

    SMS_NOTIFICATION_DURATION = Histogram(
        "sms_notification_duration_seconds",
        "Durée d'envoi d'un SMS",
        ["region"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
    )

    EMAIL_NOTIFICATION_DURATION = Histogram(
        "email_notification_duration_seconds",
        "Durée d'envoi d'un email",
        ["region"],
        buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
    )

    PROMETHEUS_AVAILABLE = True

except ImportError:
    logger.warning("[metrics] prometheus_client not installed, metrics disabled")
    PROMETHEUS_AVAILABLE = False


def record_notification_sent(
    channel: str,
    notification_type: str,
    region: str = "unknown",
) -> None:
    """Enregistre une notification envoyée avec succès.

    Args:
        channel: Canal (push, sms, email)
        notification_type: Type de notification
        region: Région active
    """
    if PROMETHEUS_AVAILABLE:
        NOTIFICATIONS_SENT.labels(
            channel=channel,
            notification_type=notification_type,
            region=region,
        ).inc()


def record_notification_failed(
    channel: str,
    notification_type: str,
    reason: str,
    region: str = "unknown",
) -> None:
    """Enregistre une notification échouée.

    Args:
        channel: Canal (push, sms, email)
        notification_type: Type de notification
        reason: Raison de l'échec
        region: Région active
    """
    if PROMETHEUS_AVAILABLE:
        NOTIFICATIONS_FAILED.labels(
            channel=channel,
            notification_type=notification_type,
            reason=reason,
            region=region,
        ).inc()


def record_notification_deduplicated(
    notification_type: str,
    region: str = "unknown",
) -> None:
    """Enregistre une notification déduplicatée.

    Args:
        notification_type: Type de notification
        region: Région active
    """
    if PROMETHEUS_AVAILABLE:
        NOTIFICATIONS_DEDUPLICATED.labels(
            notification_type=notification_type,
            region=region,
        ).inc()


def record_push_token_invalidated(
    reason: str,
    region: str = "unknown",
) -> None:
    """Enregistre un token push invalidé.

    Args:
        reason: Raison (DeviceNotRegistered, InvalidCredentials, etc.)
        region: Région active
    """
    if PROMETHEUS_AVAILABLE:
        PUSH_TOKENS_INVALIDATED.labels(
            reason=reason,
            region=region,
        ).inc()


def record_circuit_breaker_opened(
    service: str,
    region: str = "unknown",
) -> None:
    """Enregistre l'ouverture d'un circuit breaker.

    Args:
        service: Service (expo_push, twilio_sms, etc.)
        region: Région active
    """
    if PROMETHEUS_AVAILABLE:
        CIRCUIT_BREAKER_OPENED.labels(
            service=service,
            region=region,
        ).inc()


def set_circuit_breaker_state(
    service: str,
    state: str,
    region: str = "unknown",
) -> None:
    """Met à jour l'état du circuit breaker.

    Args:
        service: Service (expo_push, twilio_sms, etc.)
        state: État (CLOSED, OPEN, HALF_OPEN)
        region: Région active
    """
    if PROMETHEUS_AVAILABLE:
        state_value = {
            "CLOSED": 0,
            "OPEN": 1,
            "HALF_OPEN": 2,
        }.get(state, -1)

        CIRCUIT_BREAKER_STATE.labels(
            service=service,
            region=region,
        ).set(state_value)


def set_active_region(region: str) -> None:
    """Met à jour la région active.

    Args:
        region: Région active (eu-west-1, eu-central-1)
    """
    if PROMETHEUS_AVAILABLE:
        region_value = {
            "eu-west-1": 1,
            "eu-central-1": 2,
        }.get(region, 0)

        ACTIVE_REGION.labels(region=region).set(region_value)


def record_region_failover(from_region: str, to_region: str) -> None:
    """Enregistre un failover entre régions.

    Args:
        from_region: Région d'origine
        to_region: Région de destination
    """
    if PROMETHEUS_AVAILABLE:
        REGION_FAILOVERS.labels(
            from_region=from_region,
            to_region=to_region,
        ).inc()


def set_redis_queue_length(queue: str, length: int, region: str = "unknown") -> None:
    """Met à jour la longueur d'une queue Redis.

    Args:
        queue: Nom de la queue
        length: Longueur actuelle
        region: Région
    """
    if PROMETHEUS_AVAILABLE:
        REDIS_QUEUE_LENGTH.labels(
            queue=queue,
            region=region,
        ).set(length)


def set_kafka_consumer_lag(
    topic: str,
    partition: int,
    lag: int,
    consumer_group: str = "notifications-consumer-group",
) -> None:
    """Met à jour le lag d'un consumer Kafka.

    Args:
        topic: Topic Kafka
        partition: Numéro de partition
        lag: Lag actuel
        consumer_group: Groupe de consumers
    """
    if PROMETHEUS_AVAILABLE:
        KAFKA_CONSUMER_LAG.labels(
            topic=topic,
            partition=str(partition),
            consumer_group=consumer_group,
        ).set(lag)


def set_dlq_size(source: str, size: int, region: str = "unknown") -> None:
    """Met à jour la taille de la Dead Letter Queue.

    Args:
        source: Source (celery, kafka)
        size: Taille actuelle
        region: Région
    """
    if PROMETHEUS_AVAILABLE:
        DLQ_MESSAGES.labels(
            source=source,
            region=region,
        ).set(size)


def observe_push_duration(duration: float, region: str = "unknown") -> None:
    """Enregistre la durée d'envoi d'une notification push.

    Args:
        duration: Durée en secondes
        region: Région
    """
    if PROMETHEUS_AVAILABLE:
        PUSH_NOTIFICATION_DURATION.labels(region=region).observe(duration)


def observe_sms_duration(duration: float, region: str = "unknown") -> None:
    """Enregistre la durée d'envoi d'un SMS.

    Args:
        duration: Durée en secondes
        region: Région
    """
    if PROMETHEUS_AVAILABLE:
        SMS_NOTIFICATION_DURATION.labels(region=region).observe(duration)


def observe_email_duration(duration: float, region: str = "unknown") -> None:
    """Enregistre la durée d'envoi d'un email.

    Args:
        duration: Durée en secondes
        region: Région
    """
    if PROMETHEUS_AVAILABLE:
        EMAIL_NOTIFICATION_DURATION.labels(region=region).observe(duration)
