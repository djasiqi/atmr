# backend/services/notifications/kafka_consumer.py
"""Kafka consumer pour traiter les notifications haute performance.

Architecture:
- Consumer group pour scaling horizontal
- Commit manuel pour garantir le traitement
- Retry borné sur erreurs transitoires ; DLQ + commit si échec persistant
- Skip + commit : payload invalide, driver introuvable, pas de tokens actifs
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
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
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "")
KAFKA_SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD", "")
KAFKA_SSL_CAFILE = os.getenv("KAFKA_SSL_CAFILE", "")
KAFKA_SSL_CERTFILE = os.getenv("KAFKA_SSL_CERTFILE", "")
KAFKA_SSL_KEYFILE = os.getenv("KAFKA_SSL_KEYFILE", "")

# Retry
MAX_RETRIES = int(os.getenv("KAFKA_NOTIFICATION_MAX_RETRIES", "3"))
KAFKA_NOTIFICATION_RETRY_BACKOFF_MS = int(
    os.getenv("KAFKA_NOTIFICATION_RETRY_BACKOFF_MS", "200")
)
KAFKA_DLQ_ACK_TIMEOUT_S = float(
    os.getenv("KAFKA_NOTIFICATION_DLQ_ACK_TIMEOUT_S", "10.0")
)

try:
    from services.monitoring.prometheus import inc_notification_kafka_skip
except Exception:  # pragma: no cover

    def inc_notification_kafka_skip(*, reason: str) -> None:
        _ = reason


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


def _is_transient_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    transient_tokens = (
        "timeout",
        "temporarily",
        "leader",
        "broker",
        "connection",
        "network",
        "throttle",
        "retry",
        "unavailable",
    )
    return any(token in name or token in message for token in transient_tokens)


class NotificationConsumerSkip(Exception):
    """Skip métier : commit offset sans DLQ (driver absent, pas de tokens, etc.)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class KafkaConsumer:
    """Consumer Kafka pour traiter les notifications."""

    def __init__(self):
        super().__init__()
        self._consumer = None
        self._running = False
        self._initialized = False

        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGINT, self._handle_sigterm)

        if KAFKA_ENABLED:
            self._init_consumer()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _init_consumer(self) -> None:
        """Initialise le consumer Kafka."""
        try:
            from kafka import KafkaConsumer as KC

            from services.kafka.bootstrap_retry import run_with_kafka_bootstrap_retry

            logger.info(
                "[kafka_consumer] Initializing Kafka consumer: %s",
                KAFKA_BOOTSTRAP_SERVERS,
            )

            def _connect():
                return KC(
                    KAFKA_TOPIC_NOTIFICATIONS,
                    KAFKA_TOPIC_SMS,
                    KAFKA_TOPIC_EMAIL,
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                    group_id=KAFKA_CONSUMER_GROUP,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    key_deserializer=lambda k: k.decode("utf-8") if k else None,
                    max_poll_records=KAFKA_MAX_POLL_RECORDS,
                    session_timeout_ms=KAFKA_SESSION_TIMEOUT_MS,
                    heartbeat_interval_ms=KAFKA_HEARTBEAT_INTERVAL_MS,
                    enable_auto_commit=False,
                    auto_offset_reset="earliest",
                    **_kafka_security_config(),
                )

            self._consumer = run_with_kafka_bootstrap_retry(
                operation_label="[kafka_consumer]",
                logger=logger,
                fn=_connect,
            )

            self._initialized = True
            logger.info("[kafka_consumer] Kafka consumer initialized successfully")

        except ImportError:
            logger.error("[kafka_consumer] kafka-python not installed")
        except Exception as e:
            logger.exception(
                "[kafka_consumer] Failed to initialize Kafka consumer: %s", e
            )
            try:
                from shared.sentry_init import capture_kafka_error

                capture_kafka_error(e)
            except Exception:
                logger.debug("[kafka_consumer] sentry capture skipped", exc_info=True)

    def _trace_id_from_message(self, message: Dict[str, Any]) -> str:
        raw = message.get("trace_id")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        data = message.get("data")
        if isinstance(data, dict):
            inner = data.get("trace_id")
            if isinstance(inner, str) and inner.strip():
                return inner.strip()
        return "n/a"

    def _log_notification_consumer_skip(self, record, reason: str) -> None:
        msg = record.value if isinstance(record.value, dict) else {}
        trace_id = self._trace_id_from_message(msg)
        logger.warning(
            "notification_consumer_skip reason=%s partition=%s offset=%s trace_id=%s topic=%s",
            reason,
            record.partition,
            record.offset,
            trace_id,
            record.topic,
        )
        try:
            inc_notification_kafka_skip(reason=reason)
        except Exception:
            logger.debug("[kafka_consumer] skip metric unavailable", exc_info=True)

    def _schema_skip_reason(self, topic: str, message: Any) -> str | None:
        if not isinstance(message, dict):
            return "invalid_payload"
        if topic == KAFKA_TOPIC_NOTIFICATIONS:
            if not all(k in message for k in ("driver_id", "title", "body")):
                return "invalid_payload"
            title, body = message.get("title"), message.get("body")
            if not isinstance(title, str) or not isinstance(body, str):
                return "invalid_payload"
            did = message["driver_id"]
            if isinstance(did, bool) or not isinstance(did, int):
                try:
                    int(did)
                except (TypeError, ValueError):
                    return "invalid_payload"
            return None
        if topic == KAFKA_TOPIC_SMS:
            if not all(k in message for k in ("phone", "message")):
                return "invalid_payload"
            if not isinstance(message.get("phone"), str) or not isinstance(
                message.get("message"), str
            ):
                return "invalid_payload"
            return None
        if topic == KAFKA_TOPIC_EMAIL:
            if not all(k in message for k in ("email", "subject", "body")):
                return "invalid_payload"
            if not isinstance(message.get("email"), str) or not isinstance(
                message.get("subject"), str
            ):
                return "invalid_payload"
            if not isinstance(message.get("body"), str):
                return "invalid_payload"
            return None
        return "invalid_payload"

    def _commit_record(self) -> None:
        assert self._consumer is not None
        self._consumer.commit()

    def _handle_record(self, record) -> None:
        assert self._consumer is not None
        message = record.value
        schema_reason = self._schema_skip_reason(record.topic, message)
        if schema_reason:
            self._log_notification_consumer_skip(record, schema_reason)
            self._commit_record()
            return

        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                self._process_message(record)
                self._commit_record()
                return
            except NotificationConsumerSkip as skip:
                self._log_notification_consumer_skip(record, skip.reason)
                self._commit_record()
                return
            except Exception as e:
                last_exc = e
                transient = _is_transient_error(e)
                if transient and attempt < MAX_RETRIES:
                    sleep_s = (KAFKA_NOTIFICATION_RETRY_BACKOFF_MS * attempt) / 1000.0
                    logger.warning(
                        "[kafka_consumer] transient error attempt=%s/%s sleep=%.3fs: %s",
                        attempt,
                        MAX_RETRIES,
                        sleep_s,
                        e,
                    )
                    time.sleep(sleep_s)
                    continue
                break

        if last_exc is not None:
            if self._send_to_dlq(record, error=str(last_exc)):
                self._commit_record()
            else:
                logger.error(
                    "[kafka_consumer] DLQ failed, offset not committed topic=%s partition=%s offset=%s",
                    record.topic,
                    record.partition,
                    record.offset,
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
                assert self._consumer is not None
                messages = self._consumer.poll(timeout_ms=1000)

                for _topic_partition, records in messages.items():
                    for record in records:
                        try:
                            self._handle_record(record)
                        except Exception:
                            logger.exception(
                                "[kafka_consumer] Unexpected failure handling record"
                            )

        except KeyboardInterrupt:
            logger.info("[kafka_consumer] Received keyboard interrupt")
        except Exception as e:
            from shared.sentry_init import capture_kafka_error, is_kafka_connection_error

            if is_kafka_connection_error(e):
                capture_kafka_error(e)
            logger.exception("[kafka_consumer] Consumer loop failed: %s", e)
            raise
        finally:
            self._shutdown()

    def _process_message(self, record) -> None:
        """Traite un message Kafka."""
        topic = record.topic
        message = record.value

        logger.debug(
            "[kafka_consumer] Processing message from %s: driver_id=%s",
            topic,
            message.get("driver_id") if isinstance(message, dict) else None,
        )

        if topic == KAFKA_TOPIC_NOTIFICATIONS:
            self._process_push_notification(message)
        elif topic == KAFKA_TOPIC_SMS:
            self._process_sms_notification(message)
        elif topic == KAFKA_TOPIC_EMAIL:
            self._process_email_notification(message)
        else:
            logger.warning("[kafka_consumer] Unknown topic: %s", topic)

    def _coerce_driver_id(self, raw: Any) -> int:
        if isinstance(raw, bool):
            raise NotificationConsumerSkip("invalid_payload")
        if isinstance(raw, int):
            return raw
        try:
            return int(raw)
        except (TypeError, ValueError) as e:
            raise NotificationConsumerSkip("invalid_payload") from e

    def _process_push_notification(self, message: Dict[str, Any]) -> None:
        """Traite une notification push."""
        from ext import db
        from models import DeviceToken, Driver
        from services.notifications.device_token_lifecycle import (
            apply_push_result_to_device_token,
            deactivate_stale_device_tokens,
        )
        from services.notifications.notification_pipeline_observability import (
            NotificationE2ETimer,
            claim_idempotency_key,
            log_notification_pipeline_event,
        )
        from services.notifications.push import send_push_message

        driver_id = self._coerce_driver_id(message["driver_id"])
        title = message["title"]
        body = message["body"]
        data = message.get("data", {})
        if not isinstance(data, dict):
            data = {}

        notification_type = message.get("notification_type", "unknown")
        notification_id = message.get("notification_id") or data.get("event_id")
        booking_id = message.get("booking_id") or data.get("booking_id")
        correlation_id = message.get("correlation_id") or data.get("correlation_id")

        log_notification_pipeline_event(
            "notification_kafka_consumed",
            notification_id=notification_id,
            booking_id=booking_id,
            driver_id=driver_id,
            notification_type=notification_type,
            correlation_id=correlation_id,
        )

        idempotency_key = message.get("idempotency_key")
        if idempotency_key and not claim_idempotency_key(str(idempotency_key)):
            log_notification_pipeline_event(
                "notification_skipped",
                notification_id=notification_id,
                booking_id=booking_id,
                driver_id=driver_id,
                notification_type=notification_type,
                correlation_id=correlation_id,
                reason="idempotency_replay",
            )
            return

        timer = NotificationE2ETimer()
        timer.start()

        driver = db.session.get(Driver, driver_id)
        if not driver:
            raise NotificationConsumerSkip("driver_not_found")

        device_tokens = DeviceToken.query.filter_by(
            driver_id=driver_id,
            is_active=True,
        ).all()

        if not device_tokens:
            raise NotificationConsumerSkip("no_active_tokens")

        success_count = 0
        invalid_count = 0
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
                correlation_id=correlation_id,
            )
            last_result = result
            apply_push_result_to_device_token(device_token.id, result)

            if result.get("ok"):
                success_count += 1
                log_notification_pipeline_event(
                    "notification_fcm_sent",
                    notification_id=notification_id,
                    booking_id=booking_id,
                    driver_id=driver_id,
                    notification_type=notification_type,
                    correlation_id=correlation_id,
                    device_token_id=device_token.id,
                )
            elif result.get("token_invalid"):
                invalid_count += 1
            else:
                log_notification_pipeline_event(
                    "notification_fcm_failed",
                    notification_id=notification_id,
                    booking_id=booking_id,
                    driver_id=driver_id,
                    notification_type=notification_type,
                    correlation_id=correlation_id,
                    error=result.get("error"),
                )

        try:
            db.session.commit()
        except Exception:
            logger.exception("[kafka_consumer] commit after push lifecycle")

        try:
            deactivate_stale_device_tokens(session=db.session)
            db.session.commit()
        except Exception:
            logger.exception("[kafka_consumer] stale token cleanup failed")

        if success_count == 0 and invalid_count == len(device_tokens):
            log_notification_pipeline_event(
                "notification_skipped",
                notification_id=notification_id,
                booking_id=booking_id,
                driver_id=driver_id,
                notification_type=notification_type,
                correlation_id=correlation_id,
                reason="all_tokens_invalid",
            )
            raise NotificationConsumerSkip("all_tokens_invalid")

        if success_count == 0:
            error_msg = (
                last_result.get("error", "Unknown error")
                if last_result
                else "No active tokens"
            )
            raise Exception(error_msg)

        timer.observe_if_started()

        logger.info(
            "[kafka_consumer] Push sent successfully to driver %s (%d/%d devices)",
            driver_id,
            success_count,
            len(device_tokens),
        )

    def _process_sms_notification(self, message: Dict[str, Any]) -> None:
        """Traite une notification SMS."""
        from services.notifications.sms import send_sms_notification

        result = send_sms_notification(
            phone=message["phone"],
            message=message["message"],
            notification_type=message.get("notification_type", "unknown"),
        )

        if not result.get("ok"):
            raise Exception(result.get("error"))

    def _process_email_notification(self, message: Dict[str, Any]) -> None:
        """Traite une notification Email."""
        from services.notifications.email import send_email_notification

        result = send_email_notification(
            email=message["email"],
            subject=message["subject"],
            body=message["body"],
            notification_type=message.get("notification_type", "unknown"),
        )

        if not result.get("ok"):
            raise Exception(result.get("error"))

    def _send_to_dlq(self, record, error: str) -> bool:
        """Envoie un message en DLQ ; retourne True si accusé réception broker obtenu."""
        try:
            import time as time_mod

            from services.notifications.kafka_producer import kafka_producer

            if kafka_producer._producer is None:
                logger.error("[kafka_consumer] DLQ skipped: producer not initialized")
                return False

            dlq_message = {
                "original_topic": record.topic,
                "original_partition": record.partition,
                "original_offset": record.offset,
                "original_message": record.value,
                "error": error,
                "timestamp": int(time_mod.time() * 1000),
            }

            future = kafka_producer._producer.send(
                KAFKA_TOPIC_DLQ,
                value=dlq_message,
            )
            future.get(timeout=KAFKA_DLQ_ACK_TIMEOUT_S)

            logger.warning(
                "[kafka_consumer] Message sent to DLQ: topic=%s, offset=%s",
                record.topic,
                record.offset,
            )
            return True

        except Exception as e:
            logger.error("[kafka_consumer] Failed to send to DLQ: %s", e)
            return False

    def _handle_sigterm(self, signum, _frame) -> None:
        logger.info("[kafka_consumer] Received signal %s, shutting down...", signum)
        self._running = False

    def _shutdown(self) -> None:
        if self._consumer:
            try:
                logger.info("[kafka_consumer] Closing consumer...")
                self._consumer.close()
                logger.info("[kafka_consumer] Consumer closed successfully")
            except Exception as e:
                logger.error("[kafka_consumer] Error closing consumer: %s", e)


def run_kafka_consumer() -> None:
    """Point d'entrée pour lancer le consumer Kafka."""
    from shared.sentry_init import init_sentry

    init_sentry()
    if not KAFKA_ENABLED:
        logger.info("[kafka_consumer] désactivé (KAFKA_ENABLED=false), sortie propre")
        sys.exit(0)

    consumer = KafkaConsumer()
    if not consumer.initialized:
        logger.error("[kafka_consumer] exiting (consumer not initialized)")
        sys.exit(1)
    consumer.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_kafka_consumer()
