# backend/services/notifications/kafka_consumer.py
"""Kafka consumer pour traiter les notifications haute performance.

Architecture:
- Consumer group pour scaling horizontal
- Commit manuel pour garantir le traitement
- Retry automatique avec DLQ
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Configuration Kafka
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "notifications-consumer-group")
KAFKA_TOPIC_NOTIFICATIONS = os.getenv("KAFKA_TOPIC_NOTIFICATIONS", "notifications.push")
KAFKA_TOPIC_SMS = os.getenv("KAFKA_TOPIC_SMS", "notifications.sms")
KAFKA_TOPIC_EMAIL = os.getenv("KAFKA_TOPIC_EMAIL", "notifications.email")
KAFKA_TOPIC_DLQ = os.getenv("KAFKA_TOPIC_DLQ", "notifications.dlq")

# Performance
KAFKA_MAX_POLL_RECORDS = int(os.getenv("KAFKA_MAX_POLL_RECORDS", "500"))
KAFKA_SESSION_TIMEOUT_MS = int(os.getenv("KAFKA_SESSION_TIMEOUT_MS", "30000"))
KAFKA_HEARTBEAT_INTERVAL_MS = int(os.getenv("KAFKA_HEARTBEAT_INTERVAL_MS", "10000"))

# Retry
MAX_RETRIES = 3


class KafkaConsumer:
    """Consumer Kafka pour traiter les notifications."""

    def __init__(self):
        super().__init__()
        self._consumer = None
        self._running = False
        self._initialized = False

        # Intercepter SIGTERM pour graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

        if KAFKA_ENABLED:
            self._init_consumer()

    def _init_consumer(self) -> None:
        """Initialise le consumer Kafka."""
        try:
            from kafka import KafkaConsumer as KC  # type: ignore

            logger.info(
                "[kafka_consumer] Initializing Kafka consumer: %s",
                KAFKA_BOOTSTRAP_SERVERS,
            )

            self._consumer = KC(
                KAFKA_TOPIC_NOTIFICATIONS,
                KAFKA_TOPIC_SMS,
                KAFKA_TOPIC_EMAIL,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                group_id=KAFKA_CONSUMER_GROUP,
                # Désérialisation JSON
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                key_deserializer=lambda k: k.decode("utf-8") if k else None,
                # Performance
                max_poll_records=KAFKA_MAX_POLL_RECORDS,
                session_timeout_ms=KAFKA_SESSION_TIMEOUT_MS,
                heartbeat_interval_ms=KAFKA_HEARTBEAT_INTERVAL_MS,
                # Commit manuel pour garantir traitement
                enable_auto_commit=False,
                # Start from beginning si nouveau consumer group
                auto_offset_reset="earliest",
            )

            self._initialized = True
            logger.info("[kafka_consumer] Kafka consumer initialized successfully")

        except ImportError:
            logger.error("[kafka_consumer] kafka-python not installed")
        except Exception as e:
            logger.exception(
                "[kafka_consumer] Failed to initialize Kafka consumer: %s", e
            )

    def start(self) -> None:
        """Démarre la consommation des messages."""
        if not self._initialized:
            logger.error("[kafka_consumer] Cannot start: not initialized")
            return

        self._running = True
        logger.info("[kafka_consumer] Starting consumer loop...")

        try:
            while self._running:
                # Poll messages (timeout 1s)
                assert self._consumer is not None  # Type narrowing for pyright
                messages = self._consumer.poll(timeout_ms=1000)

                for _topic_partition, records in messages.items():
                    for record in records:
                        try:
                            self._process_message(record)
                            # Commit après traitement réussi
                            assert (
                                self._consumer is not None
                            )  # Type narrowing for pyright
                            self._consumer.commit()
                        except Exception as e:
                            logger.exception(
                                "[kafka_consumer] Failed to process message: %s", e
                            )
                            # Envoyer en DLQ
                            self._send_to_dlq(record, error=str(e))

        except KeyboardInterrupt:
            logger.info("[kafka_consumer] Received keyboard interrupt")
        finally:
            self._shutdown()

    def _process_message(self, record) -> None:
        """Traite un message Kafka.

        Args:
            record: Record Kafka
        """
        topic = record.topic
        message = record.value

        logger.debug(
            "[kafka_consumer] Processing message from %s: driver_id=%s",
            topic,
            message.get("driver_id"),
        )

        # Router selon le topic
        if topic == KAFKA_TOPIC_NOTIFICATIONS:
            self._process_push_notification(message)
        elif topic == KAFKA_TOPIC_SMS:
            self._process_sms_notification(message)
        elif topic == KAFKA_TOPIC_EMAIL:
            self._process_email_notification(message)
        else:
            logger.warning("[kafka_consumer] Unknown topic: %s", topic)

    def _process_push_notification(self, message: Dict[str, Any]) -> None:
        """Traite une notification push.

        Args:
            message: Message Kafka
        """
        from ext import db
        from models import Driver
        from services.notifications.device_token_lifecycle import (
            is_push_device_token_lifecycle_enabled,
        )
        from services.notifications.push import send_push_message

        driver_id = message["driver_id"]
        title = message["title"]
        body = message["body"]
        data = message.get("data", {})

        # Récupérer le driver
        driver = db.session.get(Driver, driver_id)
        if not driver:
            logger.warning(
                "[kafka_consumer] Driver %s not found, skipping",
                driver_id,
            )
            return

        # ✅ CORRECTIF #3: Utiliser DeviceToken pour support multi-device
        from models import DeviceToken

        device_tokens = DeviceToken.query.filter_by(
            driver_id=driver_id,
            is_active=True,
        ).all()

        if not device_tokens:
            logger.debug(
                "[kafka_consumer] Driver %s has no active push tokens, skipping",
                driver_id,
            )
            return

        # Envoyer à tous les devices actifs
        success_count = 0
        last_result: Dict[str, Any] | None = None
        for device_token in device_tokens:
            result = send_push_message(
                token=device_token.token,
                title=title,
                body=body,
                data=data,
                driver_id=driver_id,
                bypass_rate_limit=message.get("bypass_rate_limit", False),
                provider=getattr(device_token, "provider", None),
                platform=getattr(device_token, "platform", None),
                device_token_id=device_token.id,
            )
            last_result = result  # Garder le dernier résultat pour logging

            if result.get("ok"):
                success_count += 1
            elif result.get("token_invalid") and not is_push_device_token_lifecycle_enabled():
                device_token.is_active = False

        try:
            db.session.commit()
        except Exception:
            logger.exception("[kafka_consumer] commit after push lifecycle")

        if success_count == 0:
            logger.warning(
                "[kafka_consumer] Push failed for all devices of driver %s",
                driver_id,
            )
            # L'exception sera catchée par le caller et envoyée en DLQ
            error_msg = (
                last_result.get("error", "Unknown error")
                if last_result
                else "No active tokens"
            )
            raise Exception(error_msg)

        logger.info(
            "[kafka_consumer] Push sent successfully to driver %s (%d/%d devices)",
            driver_id,
            success_count,
            len(device_tokens),
        )

    def _process_sms_notification(self, message: Dict[str, Any]) -> None:
        """Traite une notification SMS.

        Args:
            message: Message Kafka
        """
        from services.notifications.sms import send_sms_notification

        result = send_sms_notification(
            phone=message["phone"],
            message=message["message"],
            notification_type=message.get("notification_type", "unknown"),
        )

        if not result.get("ok"):
            raise Exception(result.get("error"))

    def _process_email_notification(self, message: Dict[str, Any]) -> None:
        """Traite une notification Email.

        Args:
            message: Message Kafka
        """
        from services.notifications.email import send_email_notification

        result = send_email_notification(
            email=message["email"],
            subject=message["subject"],
            body=message["body"],
            notification_type=message.get("notification_type", "unknown"),
        )

        if not result.get("ok"):
            raise Exception(result.get("error"))

    def _send_to_dlq(self, record, error: str) -> None:
        """Envoie un message en Dead Letter Queue.

        Args:
            record: Record Kafka original
            error: Message d'erreur
        """
        try:
            import time

            from services.notifications.kafka_producer import kafka_producer

            dlq_message = {
                "original_topic": record.topic,
                "original_partition": record.partition,
                "original_offset": record.offset,
                "original_message": record.value,
                "error": error,
                "timestamp": int(time.time() * 1000),
            }

            assert kafka_producer._producer is not None  # Type narrowing for pyright
            kafka_producer._producer.send(
                KAFKA_TOPIC_DLQ,
                value=dlq_message,
            )

            logger.warning(
                "[kafka_consumer] Message sent to DLQ: topic=%s, offset=%s",
                record.topic,
                record.offset,
            )

        except Exception as e:
            logger.error("[kafka_consumer] Failed to send to DLQ: %s", e)

    def _handle_sigterm(self, signum, _frame) -> None:
        """Gère le signal SIGTERM pour graceful shutdown.

        Args:
            signum: Numéro du signal
            frame: Frame
        """
        logger.info("[kafka_consumer] Received signal %s, shutting down...", signum)
        self._running = False

    def _shutdown(self) -> None:
        """Arrête le consumer proprement."""
        if self._consumer:
            try:
                logger.info("[kafka_consumer] Closing consumer...")
                self._consumer.close()
                logger.info("[kafka_consumer] Consumer closed successfully")
            except Exception as e:
                logger.error("[kafka_consumer] Error closing consumer: %s", e)


def run_kafka_consumer() -> None:
    """Point d'entrée pour lancer le consumer Kafka."""
    if not KAFKA_ENABLED:
        logger.error("[kafka_consumer] Kafka is disabled (KAFKA_ENABLED=false)")
        sys.exit(1)

    consumer = KafkaConsumer()
    consumer.start()


if __name__ == "__main__":
    # Configurer logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_kafka_consumer()
