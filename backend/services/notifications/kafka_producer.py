# backend/services/notifications/kafka_producer.py
"""Kafka producer pour notifications haute performance.

Avantages vs Celery/Redis:
- Throughput: 1M+ messages/sec (vs 10K pour Redis)
- Durabilité: Réplication multi-broker
- Ordering: Garantie d'ordre par partition
- Backpressure: Gestion native de la charge
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

from services.region_router import kafka_partition_key

logger = logging.getLogger(__name__)

# Configuration Kafka
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
KAFKA_TOPIC_NOTIFICATIONS = os.getenv("KAFKA_TOPIC_NOTIFICATIONS", "notifications.push")
KAFKA_TOPIC_SMS = os.getenv("KAFKA_TOPIC_SMS", "notifications.sms")
KAFKA_TOPIC_EMAIL = os.getenv("KAFKA_TOPIC_EMAIL", "notifications.email")

# Performance tuning
KAFKA_COMPRESSION_TYPE = os.getenv(
    "KAFKA_COMPRESSION_TYPE", "snappy"
)  # snappy, gzip, lz4
KAFKA_BATCH_SIZE = int(os.getenv("KAFKA_BATCH_SIZE", "16384"))  # 16KB
KAFKA_LINGER_MS = int(os.getenv("KAFKA_LINGER_MS", "10"))  # Attendre 10ms pour batcher
KAFKA_BUFFER_MEMORY = int(os.getenv("KAFKA_BUFFER_MEMORY", "33554432"))  # 32MB
KAFKA_MAX_IN_FLIGHT = int(os.getenv("KAFKA_MAX_IN_FLIGHT", "5"))
KAFKA_ACKS = os.getenv("KAFKA_ACKS", "all")  # 0=no ack, 1=leader ack, all=all replicas
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "")
KAFKA_SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD", "")
KAFKA_SSL_CAFILE = os.getenv("KAFKA_SSL_CAFILE", "")
KAFKA_SSL_CERTFILE = os.getenv("KAFKA_SSL_CERTFILE", "")
KAFKA_SSL_KEYFILE = os.getenv("KAFKA_SSL_KEYFILE", "")


def _kafka_security_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {"security_protocol": KAFKA_SECURITY_PROTOCOL}
    if KAFKA_SASL_MECHANISM:
        cfg["sasl_mechanism"] = KAFKA_SASL_MECHANISM
    if KAFKA_SASL_USERNAME:
        cfg["sasl_plain_username"] = KAFKA_SASL_USERNAME
    if KAFKA_SASL_PASSWORD:
        cfg["sasl_plain_password"] = KAFKA_SASL_PASSWORD
    if KAFKA_SSL_CAFILE:
        cfg["ssl_cafile"] = KAFKA_SSL_CAFILE
    if KAFKA_SSL_CERTFILE:
        cfg["ssl_certfile"] = KAFKA_SSL_CERTFILE
    if KAFKA_SSL_KEYFILE:
        cfg["ssl_keyfile"] = KAFKA_SSL_KEYFILE
    return cfg


class KafkaProducer:
    """Producer Kafka pour notifications haute performance."""

    def __init__(self):
        super().__init__()
        self._producer = None
        self._initialized = False

        if KAFKA_ENABLED:
            self._init_producer()

    def _init_producer(self) -> None:
        """Initialise le producer Kafka."""
        try:
            from kafka import KafkaProducer as KP

            logger.info(
                "[kafka] Initializing Kafka producer: %s",
                KAFKA_BOOTSTRAP_SERVERS,
            )

            self._producer = KP(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                # Sérialisation JSON
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                # Performance
                compression_type=KAFKA_COMPRESSION_TYPE,
                batch_size=KAFKA_BATCH_SIZE,
                linger_ms=KAFKA_LINGER_MS,
                buffer_memory=KAFKA_BUFFER_MEMORY,
                max_in_flight_requests_per_connection=KAFKA_MAX_IN_FLIGHT,
                acks=KAFKA_ACKS,
                # Retry & reliability
                retries=3,
                max_block_ms=10000,  # Timeout si queue pleine
                # Idempotence pour éviter duplications
                enable_idempotence=True,
                **_kafka_security_config(),
            )

            self._initialized = True
            logger.info("[kafka] Kafka producer initialized successfully")

        except ImportError:
            logger.error(
                "[kafka] kafka-python not installed. Install with: pip install kafka-python"
            )
        except Exception as e:
            logger.exception("[kafka] Failed to initialize Kafka producer: %s", e)

    def send_push_notification(
        self,
        driver_id: int,
        title: str,
        body: str,
        data: Dict[str, Any] | None = None,
        *,
        notification_type: str = "unknown",
        bypass_rate_limit: bool = False,
        priority: str = "default",
    ) -> bool:
        """Envoie une notification push via Kafka.

        Args:
            driver_id: ID du driver
            title: Titre de la notification
            body: Corps de la notification
            data: Données additionnelles
            notification_type: Type de notification
            bypass_rate_limit: Si True, contourne le rate limiting
            priority: Priorité (default, high)

        Returns:
            True si envoyé avec succès
        """
        if not self._initialized:
            logger.warning("[kafka] Kafka not initialized, skipping")
            return False

        try:
            from services.notifications.notification_pipeline_observability import (
                build_idempotency_key,
                log_notification_pipeline_event,
            )

            payload_data = data or {}
            idempotency_key = build_idempotency_key(
                driver_id=driver_id,
                notification_type=notification_type,
                title=title,
                body=body,
                data=payload_data,
            )
            message = {
                "driver_id": driver_id,
                "title": title,
                "body": body,
                "data": payload_data,
                "notification_type": notification_type,
                "bypass_rate_limit": bypass_rate_limit,
                "priority": priority,
                "timestamp": int(time.time() * 1000),
                "idempotency_key": idempotency_key,
                "notification_id": payload_data.get("event_id")
                or payload_data.get("trace_id"),
                "booking_id": payload_data.get("booking_id"),
                "correlation_id": payload_data.get("correlation_id"),
            }

            log_notification_pipeline_event(
                "notification_kafka_published",
                notification_id=message.get("notification_id"),
                booking_id=message.get("booking_id"),
                driver_id=driver_id,
                notification_type=notification_type,
                correlation_id=message.get("correlation_id"),
                idempotency_key=idempotency_key,
            )

            # Utiliser driver_id comme clé pour garantir l'ordre par driver
            data_region_id_obj = (data or {}).get("region_id")
            data_company_id_obj = (data or {}).get("company_id")
            region_id = (
                data_region_id_obj if isinstance(data_region_id_obj, str) else None
            )
            company_id = (
                data_company_id_obj if isinstance(data_company_id_obj, int) else None
            )
            key = kafka_partition_key(
                region_id=region_id,
                company_id=company_id,
                driver_id=driver_id,
            )

            # Envoyer de manière asynchrone
            assert self._producer is not None  # Type narrowing for pyright
            future = self._producer.send(
                KAFKA_TOPIC_NOTIFICATIONS,
                value=message,
                key=key,
            )

            # Callback pour logging
            future.add_callback(self._on_send_success, driver_id, notification_type)
            future.add_errback(self._on_send_error, driver_id, notification_type)

            logger.debug(
                "[kafka] Push notification queued for driver %s (type: %s)",
                driver_id,
                notification_type,
            )

            return True

        except Exception as e:
            logger.exception("[kafka] Failed to send push notification: %s", e)
            return False

    def send_sms_notification(
        self,
        driver_id: int,
        phone: str,
        message: str,
        notification_type: str = "unknown",
    ) -> bool:
        """Envoie une notification SMS via Kafka.

        Args:
            driver_id: ID du driver
            phone: Numéro de téléphone
            message: Contenu du SMS
            notification_type: Type de notification

        Returns:
            True si envoyé avec succès
        """
        if not self._initialized:
            logger.warning("[kafka] Kafka not initialized, skipping")
            return False

        try:
            payload = {
                "driver_id": driver_id,
                "phone": phone,
                "message": message,
                "notification_type": notification_type,
                "timestamp": int(time.time() * 1000),
            }

            key = kafka_partition_key(
                region_id=None,
                company_id=None,
                driver_id=driver_id,
            )

            assert self._producer is not None  # Type narrowing for pyright
            future = self._producer.send(
                KAFKA_TOPIC_SMS,
                value=payload,
                key=key,
            )

            future.add_callback(self._on_send_success, driver_id, "sms")
            future.add_errback(self._on_send_error, driver_id, "sms")

            logger.debug("[kafka] SMS notification queued for driver %s", driver_id)

            return True

        except Exception as e:
            logger.exception("[kafka] Failed to send SMS notification: %s", e)
            return False

    def send_email_notification(
        self,
        driver_id: int,
        email: str,
        subject: str,
        body: str,
        notification_type: str = "unknown",
    ) -> bool:
        """Envoie une notification Email via Kafka.

        Args:
            driver_id: ID du driver
            email: Adresse email
            subject: Sujet de l'email
            body: Corps de l'email
            notification_type: Type de notification

        Returns:
            True si envoyé avec succès
        """
        if not self._initialized:
            logger.warning("[kafka] Kafka not initialized, skipping")
            return False

        try:
            payload = {
                "driver_id": driver_id,
                "email": email,
                "subject": subject,
                "body": body,
                "notification_type": notification_type,
                "timestamp": int(time.time() * 1000),
            }

            key = kafka_partition_key(
                region_id=None,
                company_id=None,
                driver_id=driver_id,
            )

            assert self._producer is not None  # Type narrowing for pyright
            future = self._producer.send(
                KAFKA_TOPIC_EMAIL,
                value=payload,
                key=key,
            )

            future.add_callback(self._on_send_success, driver_id, "email")
            future.add_errback(self._on_send_error, driver_id, "email")

            logger.debug("[kafka] Email notification queued for driver %s", driver_id)

            return True

        except Exception as e:
            logger.exception("[kafka] Failed to send email notification: %s", e)
            return False

    def _on_send_success(
        self, metadata, driver_id: int, notification_type: str
    ) -> None:
        """Callback appelé après envoi réussi.

        Args:
            metadata: Métadonnées Kafka (partition, offset)
            driver_id: ID du driver
            notification_type: Type de notification
        """
        logger.debug(
            "[kafka] Message sent successfully: driver=%s, type=%s, partition=%s, offset=%s",
            driver_id,
            notification_type,
            metadata.partition,
            metadata.offset,
        )

    def _on_send_error(self, exc, driver_id: int, notification_type: str) -> None:
        """Callback appelé après erreur d'envoi.

        Args:
            exc: Exception
            driver_id: ID du driver
            notification_type: Type de notification
        """
        logger.error(
            "[kafka] Failed to send message: driver=%s, type=%s, error=%s",
            driver_id,
            notification_type,
            exc,
        )

    def flush(self, timeout: int = 10) -> None:
        """Force l'envoi de tous les messages en attente.

        Args:
            timeout: Timeout en secondes
        """
        if self._initialized:
            try:
                assert self._producer is not None  # Type narrowing for pyright
                self._producer.flush(timeout=timeout)
                logger.debug("[kafka] Producer flushed successfully")
            except Exception as e:
                logger.error("[kafka] Failed to flush producer: %s", e)

    def close(self) -> None:
        """Ferme le producer Kafka."""
        if self._initialized:
            try:
                assert self._producer is not None  # Type narrowing for pyright
                self._producer.close()
                logger.info("[kafka] Kafka producer closed")
            except Exception as e:
                logger.error("[kafka] Failed to close producer: %s", e)


# Instance globale du producer Kafka
kafka_producer = KafkaProducer()


def send_push_via_kafka(
    driver_id: int,
    title: str,
    body: str,
    data: Dict[str, Any] | None = None,
    **kwargs,
) -> bool:
    """Envoie une notification push via Kafka.

    Args:
        driver_id: ID du driver
        title: Titre
        body: Corps
        data: Données additionnelles
        **kwargs: Autres paramètres

    Returns:
        True si envoyé avec succès
    """
    return kafka_producer.send_push_notification(
        driver_id=driver_id,
        title=title,
        body=body,
        data=data,
        **kwargs,
    )


def send_sms_via_kafka(
    driver_id: int,
    phone: str,
    message: str,
    notification_type: str = "unknown",
) -> bool:
    """Envoie une notification SMS via Kafka.

    Args:
        driver_id: ID du driver
        phone: Numéro
        message: Message
        notification_type: Type

    Returns:
        True si envoyé avec succès
    """
    return kafka_producer.send_sms_notification(
        driver_id=driver_id,
        phone=phone,
        message=message,
        notification_type=notification_type,
    )


def send_email_via_kafka(
    driver_id: int,
    email: str,
    subject: str,
    body: str,
    notification_type: str = "unknown",
) -> bool:
    """Envoie une notification Email via Kafka.

    Args:
        driver_id: ID du driver
        email: Email
        subject: Sujet
        body: Corps
        notification_type: Type

    Returns:
        True si envoyé avec succès
    """
    return kafka_producer.send_email_notification(
        driver_id=driver_id,
        email=email,
        subject=subject,
        body=body,
        notification_type=notification_type,
    )
