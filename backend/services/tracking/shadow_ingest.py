"""Consumer shadow isolé — Phase 2 (aucune projection métier).

Lit ``driver.location.raw.shadow.v3`` et compare / métrique uniquement.
Interdit : ledger autoritaire, driver, Redis, outbox, trip_tracking.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from typing import Any

logger = logging.getLogger(__name__)

SHADOW_TOPIC = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_RAW_SHADOW_V3",
    "driver.location.raw.shadow.v3",
)
SHADOW_GROUP = os.getenv(
    "KAFKA_TRACKING_SHADOW_CONSUMER_GROUP",
    "tracking-shadow-compare-v3",
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)

# Ring buffer local pour comparateur (pas de write métier)
_SHADOW_EVENTS: dict[str, dict[str, Any]] = {}
_SHADOW_EVENTS_MAX = int(os.getenv("TRACKING_SHADOW_BUFFER_MAX", "5000"))


def _inc_divergence(code: str) -> None:
    try:
        from services.monitoring.driver_location_metrics import (  # type: ignore
            inc_tracking_shadow_divergence,
        )

        inc_tracking_shadow_divergence(reason=code)
    except Exception:
        logger.info("[shadow] divergence code=%s", code)


def record_shadow_event(message: dict[str, Any]) -> dict[str, str]:
    """Enregistre un event shadow en mémoire pour comparaison (zéro write PG/Redis)."""
    eid = str(
        message.get("location_event_id")
        or message.get("tracking_event_id")
        or (message.get("payload") or {}).get("location_event_id")
        or ""
    )
    if eid:
        if len(_SHADOW_EVENTS) >= _SHADOW_EVENTS_MAX:
            # Eviction FIFO simple
            oldest = next(iter(_SHADOW_EVENTS))
            _SHADOW_EVENTS.pop(oldest, None)
        _SHADOW_EVENTS[eid] = message
    logger.info(
        "[shadow] observed location_event_id=%s driver_id=%s — no authoritative write",
        eid,
        message.get("driver_id"),
    )
    return {"status": "shadow_recorded", "location_event_id": eid}


def compare_shadow_vs_direct(
    *,
    location_event_id: str,
    shadow_payload: dict[str, Any] | None,
    direct_payload: dict[str, Any] | None,
) -> str:
    """Retourne un code divergence pour ``tracking_shadow_divergence_total``."""
    if shadow_payload is None and direct_payload is not None:
        return "shadow_missing_in_kafka"
    if direct_payload is None and shadow_payload is not None:
        return "shadow_missing_in_direct"
    if shadow_payload is None and direct_payload is None:
        return "shadow_both_missing"
    assert shadow_payload is not None and direct_payload is not None
    for key in ("latitude", "longitude", "recorded_at", "company_id"):
        s_val = shadow_payload.get(key)
        d_val = direct_payload.get(key)
        if s_val is None and isinstance(shadow_payload.get("payload"), dict):
            s_val = shadow_payload["payload"].get(key)
        if d_val is None and isinstance(direct_payload.get("payload"), dict):
            d_val = direct_payload["payload"].get(key)
        if s_val != d_val:
            return "shadow_payload_mismatch"
    return "shadow_match"


def observe_direct_for_compare(message: dict[str, Any]) -> str:
    """Appelé par le chemin legacy/direct pour comparer avec le buffer shadow."""
    eid = str(
        message.get("location_event_id")
        or (message.get("payload") or {}).get("location_event_id")
        or ""
    )
    shadow = _SHADOW_EVENTS.get(eid)
    code = compare_shadow_vs_direct(
        location_event_id=eid,
        shadow_payload=shadow,
        direct_payload=message,
    )
    if code != "shadow_match":
        _inc_divergence(code)
    return code


class TrackingShadowConsumer:
    """Boucle Kafka isolée — commit après observation seule."""

    def __init__(self) -> None:
        self._consumer = None
        self._running = False
        self._initialized = False

    def initialize(self) -> bool:
        try:
            from kafka import KafkaConsumer
        except Exception:
            logger.exception("[shadow] kafka-python unavailable")
            return False
        self._consumer = KafkaConsumer(
            SHADOW_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
            group_id=SHADOW_GROUP,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        self._initialized = True
        return True

    def _commit_record(self, record: Any) -> None:
        from kafka.structs import OffsetAndMetadata, TopicPartition

        assert self._consumer is not None
        tp = TopicPartition(record.topic, record.partition)
        self._consumer.commit(
            {tp: OffsetAndMetadata(record.offset + 1, "", -1)}
        )

    def start(self) -> None:
        if not self._initialized and not self.initialize():
            raise RuntimeError("shadow consumer init failed")
        assert self._consumer is not None
        self._running = True

        def _stop(*_args: Any) -> None:
            self._running = False

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        logger.info("[shadow] start topic=%s group=%s", SHADOW_TOPIC, SHADOW_GROUP)
        while self._running:
            polled = self._consumer.poll(timeout_ms=1000)
            for _tp, records in polled.items():
                for record in records:
                    value = record.value if isinstance(record.value, dict) else {}
                    record_shadow_event(value)
                    self._commit_record(record)
            time.sleep(0.01)


def run_tracking_shadow_consumer() -> None:
    consumer = TrackingShadowConsumer()
    consumer.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tracking_shadow_consumer()
