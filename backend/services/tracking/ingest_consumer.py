"""Consumer Kafka tracking ingest robuste.

Contrat (plan v5 / Annexe A.1 — Phase 0B) :
- consume topic raw
- échec définitif => DLQ puis commit offset uniquement si DLQ ACK confirmé
- force-commit interdit par défaut (TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE=false)
- commits explicites par partition (offset+1)
- DLQ épuisée => FatalTrackingConsumerError (fail-stop)
- Phase 1 : publication processed déplacée vers outbox_publisher
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from typing import Any

from .kafka_topics import (
    TOPIC_DRIVER_LOCATION_DLQ,
    TOPIC_DRIVER_LOCATION_PROCESSED,
    TOPIC_DRIVER_LOCATION_RAW,
)


class FatalTrackingConsumerError(RuntimeError):
    """Échec non récupérable : arrêt du consumer (pas de commit silencieux)."""

    def __init__(
        self,
        message: str,
        *,
        topic: str | None = None,
        partition: int | None = None,
        offset: int | None = None,
    ) -> None:
        super().__init__(message)
        self.topic = topic
        self.partition = partition
        self.offset = offset


try:
    from services.monitoring.driver_location_metrics import (
        inc_tracking_invalid_config,
        inc_tracking_kafka_dlq_force_commit,
        inc_tracking_kafka_dlq_messages,
        inc_tracking_kafka_messages_produced,
        inc_tracking_kafka_publish_errors,
        inc_tracking_kafka_rebalance,
        observe_tracking_kafka_e2e_latency,
        set_tracking_kafka_consumer_lag,
    )
except Exception:  # pragma: no cover

    def inc_tracking_invalid_config(*, reason: str) -> None:
        _ = reason
        pass

    def inc_tracking_kafka_messages_produced(*, topic: str) -> None:
        _ = topic
        pass

    def inc_tracking_kafka_publish_errors(*, topic: str, stage: str) -> None:
        _ = (topic, stage)
        pass

    def inc_tracking_kafka_dlq_messages(*, reason: str) -> None:
        _ = reason
        pass

    def inc_tracking_kafka_dlq_force_commit(*, reason: str) -> None:
        _ = reason
        pass

    def observe_tracking_kafka_e2e_latency(*, latency_ms: float) -> None:
        _ = latency_ms
        pass

    def inc_tracking_kafka_rebalance(*, event: str) -> None:
        _ = event
        pass

    def set_tracking_kafka_consumer_lag(
        *, group: str, topic: str, partition: int | str, lag: float
    ) -> None:
        _ = (group, topic, partition, lag)
        pass


logger = logging.getLogger(__name__)

KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
TRACKING_INGEST_ASYNC_ENABLED = (
    os.getenv("TRACKING_INGEST_ASYNC_ENABLED", "false").lower() == "true"
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
KAFKA_CONSUMER_GROUP = os.getenv(
    "KAFKA_TRACKING_CONSUMER_GROUP", "tracking-ingest-consumer-group"
)
KAFKA_COMPRESSION_TYPE = os.getenv("KAFKA_COMPRESSION_TYPE", "gzip")
KAFKA_ACKS = os.getenv("KAFKA_ACKS", "all")
KAFKA_AUTO_OFFSET_RESET = os.getenv("KAFKA_AUTO_OFFSET_RESET", "earliest")
KAFKA_MAX_RETRIES = int(os.getenv("KAFKA_MAX_RETRIES", "3"))
KAFKA_RETRY_BACKOFF_MS = int(os.getenv("KAFKA_RETRY_BACKOFF_MS", "300"))
KAFKA_PUBLISH_ACK_TIMEOUT_S = float(os.getenv("KAFKA_PUBLISH_ACK_TIMEOUT_S", "2.0"))
TRACKING_INGEST_PERSIST_ENABLED = (
    os.getenv("TRACKING_INGEST_PERSIST_ENABLED", "false").lower() == "true"
)
TRACKING_INGEST_MODE = os.getenv("TRACKING_INGEST_MODE", "legacy").lower()
# Phase 1 : TX PG+outbox puis commit RAW — plus de publish processed dans ce consumer.
_TRACKING_PERSIST_WITH_OUTBOX_DEFAULT = (
    "true"
    if TRACKING_INGEST_MODE in ("kafka_primary", "kafka_primary_canary")
    else "false"
)
TRACKING_PERSIST_WITH_OUTBOX = (
    os.getenv(
        "TRACKING_PERSIST_WITH_OUTBOX", _TRACKING_PERSIST_WITH_OUTBOX_DEFAULT
    ).lower()
    == "true"
)
TRACKING_INGEST_ALLOW_REPUBLISH_ONLY = (
    os.getenv("TRACKING_INGEST_ALLOW_REPUBLISH_ONLY", "false").lower() == "true"
)
TRACKING_INGEST_SEEK_TO_END_ON_START = (
    os.getenv("TRACKING_INGEST_SEEK_TO_END_ON_START", "false").lower() == "true"
)
# P1-1a : métrique de lag consumer (garde-fou avant activation persistance).
# Désactivable à chaud (rollback < 1 min) si end_offsets() surcharge le broker.
TRACKING_KAFKA_LAG_METRIC_ENABLED = (
    os.getenv("TRACKING_KAFKA_LAG_METRIC_ENABLED", "true").lower() == "true"
)
TRACKING_KAFKA_LAG_METRIC_INTERVAL_S = float(
    os.getenv("TRACKING_KAFKA_LAG_METRIC_INTERVAL_S", "15")
)
KAFKA_MAX_BLOCK_MS = int(os.getenv("KAFKA_MAX_BLOCK_MS", "500"))
TRACKING_DLQ_RETRY_BACKOFF_S = float(os.getenv("TRACKING_DLQ_RETRY_BACKOFF_S", "1.0"))
TRACKING_DLQ_PUBLISH_MAX_ATTEMPTS = int(
    os.getenv("TRACKING_DLQ_PUBLISH_MAX_ATTEMPTS", "3")
)
TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE = (
    os.getenv("TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE", "false").lower() == "true"
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


def _header_int(value: int) -> bytes:
    return str(max(0, value)).encode("utf-8")


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
    )
    return any(token in name or token in message for token in transient_tokens)


def _raise_fail_stop(
    record,
    *,
    reason: str,
    error: BaseException,
) -> None:
    logger.error(
        "[tracking_consumer] fail-stop reason=%s err_type=%s",
        reason,
        type(error).__name__,
        exc_info=True,
    )
    raise FatalTrackingConsumerError(
        reason,
        topic=getattr(record, "topic", None),
        partition=getattr(record, "partition", None),
        offset=getattr(record, "offset", None),
    ) from error


class TrackingIngestConsumer:
    def __init__(self) -> None:
        super().__init__()
        self._consumer = None
        self._producer = None
        self._running = False
        self._initialized = False
        self._last_lag_publish_ts = 0.0
        signal.signal(signal.SIGTERM, self._shutdown_signal)
        signal.signal(signal.SIGINT, self._shutdown_signal)
        if KAFKA_ENABLED and TRACKING_INGEST_ASYNC_ENABLED:
            self._init_clients()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _init_clients(self) -> None:
        try:
            from kafka import ConsumerRebalanceListener
            from kafka import KafkaConsumer as KC
            from kafka import KafkaProducer as KP

            from services.kafka.bootstrap_retry import run_with_kafka_bootstrap_retry

            class _RebalanceListener(ConsumerRebalanceListener):
                def on_partitions_revoked(self, revoked):
                    try:
                        inc_tracking_kafka_rebalance(event="revoked")
                    except Exception:
                        logger.debug(
                            "[tracking_consumer] rebalance metric unavailable",
                            exc_info=True,
                        )
                    logger.warning("[tracking_consumer] partitions revoked=%s", revoked)

                def on_partitions_assigned(self, assigned):
                    try:
                        inc_tracking_kafka_rebalance(event="assigned")
                    except Exception:
                        logger.debug(
                            "[tracking_consumer] rebalance metric unavailable",
                            exc_info=True,
                        )
                    logger.info("[tracking_consumer] partitions assigned=%s", assigned)

            listener = _RebalanceListener()

            def _connect():
                consumer = KC(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                    group_id=KAFKA_CONSUMER_GROUP,
                    enable_auto_commit=False,
                    auto_offset_reset=KAFKA_AUTO_OFFSET_RESET,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    key_deserializer=lambda k: k.decode("utf-8") if k else None,
                    **_kafka_security_config(),
                )
                consumer.subscribe([TOPIC_DRIVER_LOCATION_RAW], listener=listener)
                producer = KP(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    key_serializer=lambda k: k.encode("utf-8") if k else None,
                    acks=KAFKA_ACKS,
                    compression_type=KAFKA_COMPRESSION_TYPE,
                    enable_idempotence=True,
                    retries=3,
                    max_block_ms=KAFKA_MAX_BLOCK_MS,
                    **_kafka_security_config(),
                )
                return consumer, producer

            self._consumer, self._producer = run_with_kafka_bootstrap_retry(
                operation_label="[tracking_consumer]",
                logger=logger,
                fn=_connect,
            )
            self._initialized = True
            logger.info("[tracking_consumer] initialized")
        except ImportError:
            logger.error("[tracking_consumer] kafka-python dependency missing")
        except Exception as exc:
            logger.exception("[tracking_consumer] initialization failed")
            try:
                from shared.sentry_init import capture_kafka_error

                capture_kafka_error(exc)
            except Exception:
                logger.debug(
                    "[tracking_consumer] sentry capture skipped", exc_info=True
                )

    def _is_valid(self, message: dict[str, Any]) -> bool:
        payload = message.get("payload")
        if not isinstance(payload, dict):
            return False
        has_lat = "latitude" in payload or "lat" in payload
        has_lon = "longitude" in payload or "lon" in payload
        return has_lat and has_lon

    def _publish_with_ack(
        self,
        *,
        topic: str,
        key: str,
        message: dict[str, Any],
        retry_count: int,
    ) -> None:
        assert self._producer is not None
        future = self._producer.send(
            topic,
            key=key,
            value=message,
            headers=[("retry_count", _header_int(retry_count))],
        )
        future.get(timeout=KAFKA_PUBLISH_ACK_TIMEOUT_S)
        try:
            inc_tracking_kafka_messages_produced(topic=topic)
        except Exception:
            logger.debug(
                "[tracking_consumer] produced metric unavailable", exc_info=True
            )

    def _commit_record(self, record) -> None:
        """Commit explicite offset+1 pour la partition du record uniquement."""
        assert self._consumer is not None
        from kafka.structs import OffsetAndMetadata, TopicPartition

        tp = TopicPartition(record.topic, record.partition)
        self._consumer.commit({tp: OffsetAndMetadata(record.offset + 1, "", -1)})

    def _commit_current(self) -> None:
        """Compat tests legacy — préférer ``_commit_record``."""
        assert self._consumer is not None
        self._consumer.commit()

    def _send_to_dlq_and_commit(
        self,
        *,
        record,
        key: str,
        source_message: dict[str, Any],
        error: Exception,
        retry_count: int,
        error_type: str,
    ) -> bool:
        dlq_payload = {
            "original_topic": record.topic,
            "original_partition": record.partition,
            "original_offset": record.offset,
            "original_key": record.key,
            "original_message": source_message,
            "error": str(error),
            "error_type": error_type,
            "retry_count": retry_count,
            "timestamp": int(time.time() * 1000),
        }
        for dlq_attempt in range(1, TRACKING_DLQ_PUBLISH_MAX_ATTEMPTS + 1):
            try:
                self._publish_with_ack(
                    topic=TOPIC_DRIVER_LOCATION_DLQ,
                    key=key,
                    message=dlq_payload,
                    retry_count=retry_count,
                )
                self._commit_record(record)
                try:
                    inc_tracking_kafka_dlq_messages(reason=error_type)
                except Exception:
                    logger.debug(
                        "[tracking_consumer] dlq metric unavailable", exc_info=True
                    )
                logger.warning(
                    "[tracking_consumer] DLQ confirmed topic=%s partition=%s offset=%s type=%s",
                    record.topic,
                    record.partition,
                    record.offset,
                    error_type,
                )
                return True
            except Exception:
                try:
                    inc_tracking_kafka_publish_errors(
                        topic=TOPIC_DRIVER_LOCATION_DLQ, stage="dlq_publish_failed"
                    )
                except Exception:
                    logger.debug(
                        "[tracking_consumer] publish error metric unavailable",
                        exc_info=True,
                    )
                logger.exception(
                    "[tracking_consumer] DLQ publish failed attempt=%s/%s topic=%s partition=%s offset=%s",
                    dlq_attempt,
                    TRACKING_DLQ_PUBLISH_MAX_ATTEMPTS,
                    record.topic,
                    record.partition,
                    record.offset,
                )
                if dlq_attempt < TRACKING_DLQ_PUBLISH_MAX_ATTEMPTS:
                    time.sleep(TRACKING_DLQ_RETRY_BACKOFF_S * dlq_attempt)
                    continue
                if TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE:
                    # Legacy chaos uniquement — interdit en prod (défaut false).
                    logger.critical(
                        "[tracking_consumer] DLQ exhausted -> force commit (LEGACY FLAG) topic=%s partition=%s offset=%s",
                        record.topic,
                        record.partition,
                        record.offset,
                    )
                    try:
                        inc_tracking_kafka_dlq_force_commit(reason=error_type)
                    except Exception:
                        logger.debug(
                            "[tracking_consumer] dlq force_commit metric unavailable",
                            exc_info=True,
                        )
                    self._commit_record(record)
                    return True
                logger.critical(
                    "[tracking_consumer] DLQ exhausted — fail-stop topic=%s partition=%s offset=%s",
                    record.topic,
                    record.partition,
                    record.offset,
                )
                raise FatalTrackingConsumerError(
                    "dlq_exhausted_no_commit",
                    topic=record.topic,
                    partition=record.partition,
                    offset=record.offset,
                ) from error
        return False

    def _observe_e2e_latency(self, message: dict[str, Any]) -> None:
        received_at_ms = message.get("received_at_ms")
        if not isinstance(received_at_ms, (int, float)):
            return
        latency_ms = max(0.0, (time.time() * 1000) - float(received_at_ms))
        try:
            observe_tracking_kafka_e2e_latency(latency_ms=latency_ms)
        except Exception:
            logger.debug("[tracking_consumer] e2e metric unavailable", exc_info=True)

    def _process_record(self, record) -> bool:
        message_obj = record.value if isinstance(record.value, dict) else {}
        try:
            driver_id_obj = message_obj.get("driver_id")
            if not isinstance(driver_id_obj, (int, str)):
                raise ValueError("driver_id_missing")
            driver_id = int(driver_id_obj)
            key = f"driver_{driver_id}"
        except Exception as exc:
            return self._send_to_dlq_and_commit(
                record=record,
                key="driver_unknown",
                source_message=message_obj,
                error=exc,
                retry_count=0,
                error_type="invalid_driver_id",
            )

        if not self._is_valid(message_obj):
            return self._send_to_dlq_and_commit(
                record=record,
                key=key,
                source_message=message_obj,
                error=ValueError("invalid_payload"),
                retry_count=0,
                error_type="invalid_payload",
            )

        for attempt in range(1, KAFKA_MAX_RETRIES + 1):
            try:
                validated = {**message_obj, "validated_at_ms": record.timestamp}
                _msg_source = str(message_obj.get("source") or "")

                if TRACKING_PERSIST_WITH_OUTBOX and _msg_source != "socket_batch":
                    # Annexe A.1 : TX PG+outbox → COMMIT offset RAW (pas de publish processed)
                    from services.tracking.persist_kafka_outbox import (
                        PersistKafkaOutboxError,
                        persist_driver_location_with_outbox_from_kafka,
                    )

                    try:
                        validated, persist_result = (
                            persist_driver_location_with_outbox_from_kafka(
                                message_obj,
                                driver_id=driver_id,
                            )
                        )
                    except PersistKafkaOutboxError as persist_exc:
                        # Erreurs de contrat session → DLQ (payload / session invalide)
                        return self._send_to_dlq_and_commit(
                            record=record,
                            key=key,
                            source_message=message_obj,
                            error=persist_exc,
                            retry_count=0,
                            error_type=persist_exc.code,
                        )
                    validated["validated_at_ms"] = record.timestamp
                    try:
                        from services.monitoring.driver_location_metrics import (
                            inc_received,
                            inc_tracking_kafka_persist,
                        )

                        payload_for_mode = validated.get("payload")
                        location_mode = (
                            str(payload_for_mode.get("location_mode"))
                            if isinstance(payload_for_mode, dict)
                            and payload_for_mode.get("location_mode")
                            else "mission_live"
                        )
                        if persist_result.get("status") != "duplicate":
                            inc_received(transport="kafka", location_mode=location_mode)
                        inc_tracking_kafka_persist(
                            accept_status=str(
                                persist_result.get("status") or "persisted"
                            ),
                        )
                    except Exception:
                        logger.debug(
                            "[tracking_consumer] persist metrics unavailable",
                            exc_info=True,
                        )
                    self._commit_record(record)
                    self._observe_e2e_latency(message_obj)
                    try:
                        from services.tracking.async_circuit import (
                            mark_consumer_persist_success,
                        )

                        mark_consumer_persist_success()
                    except Exception:
                        logger.debug(
                            "[tracking_consumer] persist heartbeat failed",
                            exc_info=True,
                        )
                    return True

                # Legacy : persist use-case + publish processed (avant bascule Phase 1)
                if TRACKING_INGEST_PERSIST_ENABLED and _msg_source != "socket_batch":
                    from services.tracking.ingest_persist import (
                        persist_driver_location_from_kafka,
                    )

                    validated, uc_result = persist_driver_location_from_kafka(
                        message_obj,
                        driver_id=driver_id,
                    )
                    validated["validated_at_ms"] = record.timestamp
                    try:
                        from services.monitoring.driver_location_metrics import (
                            inc_received,
                            inc_tracking_kafka_persist,
                        )

                        payload_for_mode = validated.get("payload")
                        location_mode = (
                            str(payload_for_mode.get("location_mode"))
                            if isinstance(payload_for_mode, dict)
                            and payload_for_mode.get("location_mode")
                            else "mission_live"
                        )
                        if not uc_result.dedup_skipped:
                            inc_received(transport="kafka", location_mode=location_mode)
                        inc_tracking_kafka_persist(
                            accept_status=uc_result.accept_status,
                        )
                    except Exception:
                        logger.debug(
                            "[tracking_consumer] persist metrics unavailable",
                            exc_info=True,
                        )
                self._publish_with_ack(
                    topic=TOPIC_DRIVER_LOCATION_PROCESSED,
                    key=key,
                    message=validated,
                    retry_count=attempt - 1,
                )
                self._commit_record(record)
                self._observe_e2e_latency(message_obj)
                return True
            except Exception as exc:
                from services.tracking.db_error_classification import (
                    DbErrorAction,
                    classify_db_error,
                )

                db_action = classify_db_error(exc)
                try:
                    inc_tracking_kafka_publish_errors(
                        topic=TOPIC_DRIVER_LOCATION_PROCESSED,
                        stage=(
                            "outbox_persist_failed"
                            if TRACKING_PERSIST_WITH_OUTBOX
                            else "processed_publish_failed"
                        ),
                    )
                except Exception:
                    logger.debug(
                        "[tracking_consumer] publish error metric unavailable",
                        exc_info=True,
                    )

                if db_action == DbErrorAction.FAIL_STOP:
                    _raise_fail_stop(
                        record,
                        reason="db_fail_stop",
                        error=exc,
                    )

                if db_action == DbErrorAction.DLQ:
                    return self._send_to_dlq_and_commit(
                        record=record,
                        key=key,
                        source_message=message_obj,
                        error=exc,
                        retry_count=attempt,
                        error_type="db_data_error",
                    )

                if db_action == DbErrorAction.INFRASTRUCTURE_RETRY:
                    if attempt < KAFKA_MAX_RETRIES:
                        sleep_s = (KAFKA_RETRY_BACKOFF_MS * attempt) / 1000.0
                        logger.warning(
                            "[tracking_consumer] infra DB error attempt=%s/%s sleep=%.3fs",
                            attempt,
                            KAFKA_MAX_RETRIES,
                            sleep_s,
                        )
                        time.sleep(sleep_s)
                        continue
                    _raise_fail_stop(
                        record,
                        reason="db_infrastructure_exhausted",
                        error=exc,
                    )

                # Hors allowlist (payload / DataError connue / PersistKafkaOutboxError) :
                # transient Kafka ou erreur inconnue → retry puis fail-stop (jamais DLQ+commit).
                transient = _is_transient_error(exc)
                if transient and attempt < KAFKA_MAX_RETRIES:
                    sleep_s = (KAFKA_RETRY_BACKOFF_MS * attempt) / 1000.0
                    logger.warning(
                        "[tracking_consumer] transient publish error attempt=%s/%s sleep=%.3fs",
                        attempt,
                        KAFKA_MAX_RETRIES,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                    continue
                _raise_fail_stop(
                    record,
                    reason=(
                        "processed_publish_exhausted"
                        if transient
                        else "unclassified_error_fail_stop"
                    ),
                    error=exc,
                )
        return False

    def _seek_to_end_on_start(self) -> None:
        """Option R7 : ignorer le backlog raw au premier démarrage avec persist."""
        assert self._consumer is not None
        logger.warning(
            "[tracking_consumer] TRACKING_INGEST_SEEK_TO_END_ON_START=true — skip backlog raw"
        )
        deadline = time.monotonic() + 30.0
        while self._running and time.monotonic() < deadline:
            self._consumer.poll(timeout_ms=500)
            assignment = self._consumer.assignment()
            if assignment:
                self._consumer.seek_to_end(*assignment)
                logger.info(
                    "[tracking_consumer] seek_to_end partitions=%s", len(assignment)
                )
                return
        logger.error(
            "[tracking_consumer] seek_to_end timeout — no partition assignment"
        )

    def start(self) -> None:
        if not self._initialized:
            logger.error("[tracking_consumer] cannot start, not initialized")
            return
        assert self._consumer is not None
        self._running = True
        if TRACKING_INGEST_SEEK_TO_END_ON_START:
            self._seek_to_end_on_start()
        logger.info("[tracking_consumer] start loop")
        try:
            while self._running:
                polled = self._consumer.poll(timeout_ms=1000)
                try:
                    from services.tracking.async_circuit import (
                        evaluate_and_store_circuit,
                        write_consumer_heartbeat,
                    )

                    write_consumer_heartbeat()
                    evaluate_and_store_circuit()
                except Exception:
                    logger.debug(
                        "[tracking_consumer] heartbeat/circuit write failed",
                        exc_info=True,
                    )
                for _tp, records in polled.items():
                    for record in records:
                        try:
                            ok = self._process_record(record)
                            if ok is False:
                                raise FatalTrackingConsumerError(
                                    "process_record_returned_false",
                                    topic=record.topic,
                                    partition=record.partition,
                                    offset=record.offset,
                                )
                        except FatalTrackingConsumerError:
                            logger.critical(
                                "[tracking_consumer] fatal error — stopping consumer",
                                exc_info=True,
                            )
                            self._running = False
                            raise
                        except Exception as exc:
                            logger.exception("[tracking_consumer] processing error")
                            raise FatalTrackingConsumerError(
                                "unexpected_processing_error",
                                topic=getattr(record, "topic", None),
                                partition=getattr(record, "partition", None),
                                offset=getattr(record, "offset", None),
                            ) from exc
                self._maybe_publish_lag()
        except FatalTrackingConsumerError:
            raise
        except Exception as exc:
            from shared.sentry_init import (
                capture_kafka_error,
                is_kafka_connection_error,
            )

            if is_kafka_connection_error(exc):
                capture_kafka_error(exc)
            logger.exception("[tracking_consumer] poll loop failed")
            raise
        finally:
            self.close()

    def _maybe_publish_lag(self) -> None:
        """Publie le lag consumer par partition (P1-1a), throttlé ~15 s.

        ``lag = end_offset - position`` (lag « prêt à traiter », évite le RPC
        ``committed()``). Garde-fou de saturation avant activation persistance.
        Best-effort : toute erreur est avalée pour ne jamais casser la boucle poll.
        """
        if not TRACKING_KAFKA_LAG_METRIC_ENABLED or self._consumer is None:
            return
        now = time.time()
        if now - self._last_lag_publish_ts < TRACKING_KAFKA_LAG_METRIC_INTERVAL_S:
            return
        self._last_lag_publish_ts = now
        try:
            assignment = self._consumer.assignment()
            if not assignment:
                return
            end_offsets = self._consumer.end_offsets(list(assignment))
            for tp in assignment:
                position = self._consumer.position(tp)
                end_offset = end_offsets.get(tp)
                if position is None or end_offset is None:
                    continue
                set_tracking_kafka_consumer_lag(
                    group=KAFKA_CONSUMER_GROUP,
                    topic=tp.topic,
                    partition=tp.partition,
                    lag=end_offset - position,
                )
        except Exception:
            logger.debug("[tracking_consumer] lag metric unavailable", exc_info=True)

    def _shutdown_signal(self, signum, _frame) -> None:
        logger.info("[tracking_consumer] shutdown signal=%s", signum)
        self._running = False

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()
        if self._producer is not None:
            self._producer.flush(timeout=3)
            self._producer.close()


def run_tracking_ingest_consumer() -> None:
    from services.monitoring.standalone_prometheus_server import (
        start_standalone_prometheus_server,
    )
    from shared.logging_utils import configure_kafka_log_noise
    from shared.sentry_init import init_sentry

    configure_kafka_log_noise()
    init_sentry()
    start_standalone_prometheus_server()
    if not KAFKA_ENABLED or not TRACKING_INGEST_ASYNC_ENABLED:
        logger.info(
            "[tracking_consumer] disabled (KAFKA_ENABLED or TRACKING_INGEST_ASYNC_ENABLED), exiting cleanly"
        )
        sys.exit(0)
    if (
        TRACKING_INGEST_ASYNC_ENABLED
        and not TRACKING_INGEST_PERSIST_ENABLED
        and not TRACKING_INGEST_ALLOW_REPUBLISH_ONLY
    ):
        logger.critical(
            "[tracking_consumer] CONFIG INVALIDE: TRACKING_INGEST_ASYNC_ENABLED=true "
            "mais TRACKING_INGEST_PERSIST_ENABLED=false -> positions enqueue Kafka "
            "jamais persistees en DB. Definir TRACKING_INGEST_PERSIST_ENABLED=true, "
            "ou TRACKING_INGEST_ALLOW_REPUBLISH_ONLY=true si rollback intentionnel "
            "(cf. docs/ops/gps-tracking-pipeline.md). Refusing to start."
        )
        try:
            inc_tracking_invalid_config(reason="async_without_persist")
        except Exception:
            logger.debug(
                "[tracking_consumer] invalid_config metric unavailable", exc_info=True
            )
        sys.exit(1)
    if (
        TRACKING_INGEST_ASYNC_ENABLED
        and not TRACKING_INGEST_PERSIST_ENABLED
        and TRACKING_INGEST_ALLOW_REPUBLISH_ONLY
    ):
        logger.warning(
            "[tracking_consumer] MODE REPUBLISH-ONLY: TRACKING_INGEST_PERSIST_ENABLED=false "
            "avec TRACKING_INGEST_ALLOW_REPUBLISH_ONLY=true — positions Kafka republiees "
            "sans ecriture DB (rollback intentionnel)."
        )
    from services.tracking.worker_bootstrap import validate_tracking_worker_env

    validate_tracking_worker_env("tracking_consumer")
    consumer = TrackingIngestConsumer()
    if not consumer.initialized:
        logger.error("[tracking_consumer] exiting (kafka clients not initialized)")
        sys.exit(1)
    consumer.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    run_tracking_ingest_consumer()
