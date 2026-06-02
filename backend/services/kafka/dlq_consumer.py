"""Consumer Kafka minimal pour DLQ (persist + alert, sans auto-replay)."""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
KAFKA_DLQ_CONSUMER_GROUP = os.getenv(
    "KAFKA_DLQ_CONSUMER_GROUP", "kafka-dlq-consumer-group"
)
KAFKA_TOPIC_NOTIFICATIONS_DLQ = os.getenv(
    "KAFKA_TOPIC_NOTIFICATIONS_DLQ", "notifications.dlq"
)
KAFKA_TOPIC_DRIVER_LOCATION_DLQ = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_DLQ", "driver.location.dlq"
)
KAFKA_DLQ_STORAGE_PATH = os.getenv(
    "KAFKA_DLQ_STORAGE_PATH", "/app/data/kafka_dlq_events.jsonl"
)

KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "")
KAFKA_SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD", "")
KAFKA_SSL_CAFILE = os.getenv("KAFKA_SSL_CAFILE", "")
KAFKA_SSL_CERTFILE = os.getenv("KAFKA_SSL_CERTFILE", "")
KAFKA_SSL_KEYFILE = os.getenv("KAFKA_SSL_KEYFILE", "")


def _kafka_security_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {"security_protocol": KAFKA_SECURITY_PROTOCOL}
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


class KafkaDlqConsumer:
    """Lit les DLQ Kafka et persiste pour investigation manuelle.

    Important: ce worker ne rejoue jamais automatiquement le métier.
    """

    def __init__(self) -> None:
        super().__init__()
        self._consumer = None
        self._running = False
        self._initialized = False
        self._persist_path = Path(KAFKA_DLQ_STORAGE_PATH)
        signal.signal(signal.SIGTERM, self._shutdown_signal)
        signal.signal(signal.SIGINT, self._shutdown_signal)
        if KAFKA_ENABLED:
            self._init_consumer()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _init_consumer(self) -> None:
        try:
            from kafka import KafkaConsumer as KC

            from services.kafka.bootstrap_retry import run_with_kafka_bootstrap_retry

            def _connect():
                return KC(
                    KAFKA_TOPIC_NOTIFICATIONS_DLQ,
                    KAFKA_TOPIC_DRIVER_LOCATION_DLQ,
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                    group_id=KAFKA_DLQ_CONSUMER_GROUP,
                    enable_auto_commit=False,
                    auto_offset_reset="earliest",
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    key_deserializer=lambda k: k.decode("utf-8") if k else None,
                    **_kafka_security_config(),
                )

            self._consumer = run_with_kafka_bootstrap_retry(
                operation_label="[kafka_dlq]",
                logger=logger,
                fn=_connect,
            )
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._initialized = True
            logger.info(
                "[kafka_dlq] initialized topics=(%s,%s) group=%s storage=%s",
                KAFKA_TOPIC_NOTIFICATIONS_DLQ,
                KAFKA_TOPIC_DRIVER_LOCATION_DLQ,
                KAFKA_DLQ_CONSUMER_GROUP,
                self._persist_path,
            )
        except ImportError:
            logger.error("[kafka_dlq] kafka-python dependency missing")
        except Exception as exc:
            logger.exception("[kafka_dlq] initialization failed")
            try:
                from shared.sentry_init import capture_kafka_error

                capture_kafka_error(exc)
            except Exception:
                logger.debug("[kafka_dlq] sentry capture skipped", exc_info=True)

    def _persist_event(self, record) -> None:
        payload = {
            "stored_at_ms": int(time.time() * 1000),
            "topic": record.topic,
            "partition": record.partition,
            "offset": record.offset,
            "key": record.key,
            "value": record.value,
        }
        with self._persist_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _update_dlq_metric(self, topic: str) -> None:
        try:
            from kafka import TopicPartition
            from services.notifications.metrics import set_dlq_size

            assert self._consumer is not None
            partitions = self._consumer.partitions_for_topic(topic) or set()
            total_lag = 0
            for partition_id in partitions:
                tp = TopicPartition(topic, partition_id)
                end_offsets = self._consumer.end_offsets([tp])
                end_offset = end_offsets.get(tp, 0)
                committed = self._consumer.committed(tp) or 0
                total_lag += max(0, end_offset - committed)
            set_dlq_size(source=topic, size=total_lag, region="global")
        except Exception:
            logger.debug("[kafka_dlq] dlq metric update skipped", exc_info=True)

    def start(self) -> None:
        if not self._initialized:
            logger.error("[kafka_dlq] cannot start, not initialized")
            return
        assert self._consumer is not None
        self._running = True
        logger.info("[kafka_dlq] start loop")
        try:
            while self._running:
                polled = self._consumer.poll(timeout_ms=1000)
                for _tp, records in polled.items():
                    for record in records:
                        try:
                            self._persist_event(record)
                            self._consumer.commit()
                            self._update_dlq_metric(record.topic)
                        except Exception:
                            logger.exception(
                                "[kafka_dlq] persist failed topic=%s partition=%s offset=%s",
                                record.topic,
                                record.partition,
                                record.offset,
                            )
        except Exception as exc:
            from shared.sentry_init import capture_kafka_error, is_kafka_connection_error

            if is_kafka_connection_error(exc):
                capture_kafka_error(exc)
            logger.exception("[kafka_dlq] poll loop failed")
            raise
        finally:
            self.close()

    def _shutdown_signal(self, signum, _frame) -> None:
        logger.info("[kafka_dlq] shutdown signal=%s", signum)
        self._running = False

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()


def run_kafka_dlq_consumer() -> None:
    from shared.sentry_init import init_sentry

    init_sentry()
    if not KAFKA_ENABLED:
        logger.info("[kafka_dlq] disabled (KAFKA_ENABLED=false), exiting cleanly")
        sys.exit(0)
    consumer = KafkaDlqConsumer()
    if not consumer.initialized:
        logger.error("[kafka_dlq] exiting (kafka consumer not initialized)")
        sys.exit(1)
    consumer.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    run_kafka_dlq_consumer()
