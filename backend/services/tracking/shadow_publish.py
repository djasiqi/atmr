"""Publication observations shadow (direct.observed + raw.shadow ACK)."""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
KAFKA_SHADOW_PUBLISH_TIMEOUT_S = float(
    os.getenv("KAFKA_SHADOW_PUBLISH_TIMEOUT_S", "2.0")
)
TRACKING_INGEST_MODE = os.getenv("TRACKING_INGEST_MODE", "legacy").strip().lower()

_producer: Any = None


def _get_producer() -> Any:
    global _producer
    if _producer is not None:
        return _producer
    from kafka import KafkaProducer

    _producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
        acks="all",
        enable_idempotence=True,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        key_serializer=lambda k: str(k).encode("utf-8"),
    )
    return _producer


def _inc_publish_failed(side: str) -> None:
    try:
        from services.monitoring.driver_location_metrics import (
            inc_tracking_shadow_publish_failed,
        )

        inc_tracking_shadow_publish_failed(side=side)
    except Exception:
        logger.info("[shadow] publish_failed side=%s", side)


def _mark_unavailable(
    *,
    driver_id: int,
    location_event_id: str,
    company_id: int | None,
    side: str,
) -> None:
    try:
        from services.tracking.shadow_store import mark_comparison_unavailable

        ok = mark_comparison_unavailable(
            driver_id=driver_id,
            location_event_id=location_event_id,
            company_id=company_id,
            side=side,
        )
        if not ok:
            logger.error(
                "[shadow] CRITICAL upsert comparison_unavailable failed "
                "side=%s driver=%s eid=%s",
                side,
                driver_id,
                location_event_id,
            )
    except Exception:
        logger.exception("[shadow] CRITICAL mark unavailable exception side=%s", side)


def publish_direct_observation(
    *,
    driver_id: int,
    company_id: int | None,
    location_event_id: str,
    payload_fingerprint: str,
    accept_status: str,
    accept_reason: str,
    persisted: bool,
) -> bool:
    """Publie direct.observed.v3 après résultat autoritaire (acks=all)."""
    if TRACKING_INGEST_MODE != "shadow_kafka":
        return False
    if not location_event_id:
        return False

    from services.tracking.kafka_topics import TOPIC_DRIVER_LOCATION_DIRECT_OBSERVED_V3

    message = {
        "driver_id": driver_id,
        "company_id": company_id,
        "location_event_id": location_event_id,
        "contract_version": 3,
        "payload_fingerprint": payload_fingerprint,
        "accept_status": accept_status,
        "accept_reason": accept_reason,
        "persisted": persisted,
        "observed_at": datetime.now(UTC).isoformat(),
    }
    try:
        producer = _get_producer()
        future = producer.send(
            TOPIC_DRIVER_LOCATION_DIRECT_OBSERVED_V3,
            key=f"driver_{driver_id}",
            value=message,
        )
        future.get(timeout=KAFKA_SHADOW_PUBLISH_TIMEOUT_S)
        return True
    except Exception:
        logger.warning(
            "[shadow] direct.observed publish failed driver=%s eid=%s",
            driver_id,
            location_event_id,
            exc_info=True,
        )
        _inc_publish_failed("direct")
        _mark_unavailable(
            driver_id=driver_id,
            location_event_id=location_event_id,
            company_id=company_id,
            side="direct",
        )
        return False


def publish_raw_shadow_copy(
    *,
    message: dict[str, Any],
    key: str,
    producer: Any | None = None,
) -> bool:
    """Copie raw.shadow.v3 avec acks=all + future.get (mode shadow_kafka)."""
    from services.tracking.kafka_topics import TOPIC_DRIVER_LOCATION_RAW_SHADOW_V3

    driver_id = int(message.get("driver_id") or 0)
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    eid = str(
        message.get("location_event_id")
        or payload.get("location_event_id")
        or payload.get("tracking_event_id")
        or ""
    )
    company_id = message.get("company_id")
    if not isinstance(company_id, int):
        company_id = (
            payload.get("company_id")
            if isinstance(payload.get("company_id"), int)
            else None
        )

    try:
        prod = producer or _get_producer()
        future = prod.send(
            TOPIC_DRIVER_LOCATION_RAW_SHADOW_V3,
            value=message,
            key=key,
        )
        future.get(timeout=KAFKA_SHADOW_PUBLISH_TIMEOUT_S)
        return True
    except Exception:
        logger.warning(
            "[shadow] raw.shadow publish failed driver=%s eid=%s",
            driver_id,
            eid,
            exc_info=True,
        )
        _inc_publish_failed("shadow")
        if driver_id and eid:
            _mark_unavailable(
                driver_id=driver_id,
                location_event_id=eid,
                company_id=company_id if isinstance(company_id, int) else None,
                side="shadow",
            )
        return False
