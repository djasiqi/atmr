"""Comparateur shadow Phase 2 — topics dual + stockage PG non autoritaire.

Consomme ``direct.observed.v3`` + ``raw.shadow.v3``, joint via
``tracking_shadow_observations``. Zéro write ledger/Redis/outbox métier.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from typing import Any

from services.tracking.shadow_evaluator import evaluate_shadow_acceptance
from services.tracking.shadow_store import (
    expire_waiting_observations,
    extend_deadlines_after_lag_recovery,
    upsert_direct_observation,
    upsert_shadow_observation,
)

logger = logging.getLogger(__name__)

SHADOW_TOPIC = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_RAW_SHADOW_V3",
    "driver.location.raw.shadow.v3",
)
DIRECT_TOPIC = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_DIRECT_OBSERVED_V3",
    "driver.location.direct.observed.v3",
)
SHADOW_GROUP = os.getenv(
    "KAFKA_TRACKING_SHADOW_CONSUMER_GROUP",
    "tracking-shadow-compare",
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
LAG_THRESHOLD_MESSAGES = int(os.getenv("TRACKING_SHADOW_LAG_THRESHOLD", "100"))


def _inc_divergence(code: str) -> None:
    try:
        from services.monitoring.driver_location_metrics import (
            inc_tracking_shadow_divergence,
        )

        inc_tracking_shadow_divergence(reason=code)
    except Exception:
        logger.info("[shadow] divergence code=%s", code)


def compare_shadow_vs_direct(
    *,
    location_event_id: str,
    shadow_payload: dict[str, Any] | None,
    direct_payload: dict[str, Any] | None,
) -> str:
    """Compat tests / codes divergence (fingerprint + acceptation)."""
    del location_event_id  # utilisé pour API stable
    if shadow_payload is None and direct_payload is not None:
        return "shadow_missing_in_kafka"
    if direct_payload is None and shadow_payload is not None:
        return "shadow_missing_in_direct"
    if shadow_payload is None and direct_payload is None:
        return "shadow_both_missing"
    assert shadow_payload is not None and direct_payload is not None

    s_fp = shadow_payload.get("payload_fingerprint") or shadow_payload.get(
        "shadow_fingerprint"
    )
    d_fp = direct_payload.get("payload_fingerprint") or direct_payload.get(
        "direct_fingerprint"
    )
    if s_fp and d_fp and s_fp != d_fp:
        return "shadow_payload_mismatch"

    s_status = shadow_payload.get("shadow_accept_status") or shadow_payload.get(
        "accept_status"
    )
    d_status = direct_payload.get("accept_status") or direct_payload.get(
        "direct_accept_status"
    )
    if s_status and d_status:
        s_ok = str(s_status).startswith("accepted")
        d_ok = str(d_status).startswith("accepted")
        if s_ok != d_ok:
            return "shadow_acceptance_mismatch"

    if s_fp and d_fp:
        return "shadow_match"

    # Fallback legacy champs GPS
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


def _metric_for_state(state: str) -> None:
    mapping = {
        "matched": "shadow_match",
        "payload_mismatch": "shadow_payload_mismatch",
        "acceptance_mismatch": "shadow_acceptance_mismatch",
        "comparison_unavailable": "comparison_unavailable",
        "expired": "expired",
        "waiting_direct": "waiting_direct",
        "waiting_shadow": "waiting_shadow",
    }
    code = mapping.get(state, state)
    if code != "shadow_match" and state not in ("waiting_direct", "waiting_shadow"):
        _inc_divergence(code)
    elif code == "shadow_match":
        _inc_divergence("shadow_match")


def handle_direct_observed(message: dict[str, Any], *, consumer_lag: int = 0) -> str:
    """Traite un message direct.observed.v3 (résultat autoritaire)."""
    driver_id = int(message.get("driver_id") or 0)
    eid = str(message.get("location_event_id") or "")
    if not driver_id or not eid:
        return "comparison_unavailable"
    company_id = message.get("company_id")
    if not isinstance(company_id, int):
        company_id = None
    state = upsert_direct_observation(
        driver_id=driver_id,
        location_event_id=eid,
        company_id=company_id,
        fingerprint=str(message.get("payload_fingerprint") or ""),
        accept_status=str(message.get("accept_status") or ""),
        accept_reason=str(message.get("accept_reason") or ""),
        consumer_lag=consumer_lag,
    )
    _metric_for_state(state)
    return state


def handle_shadow_raw(message: dict[str, Any], *, consumer_lag: int = 0) -> str:
    """Évalue pur + upsert côté shadow."""
    evaluation = evaluate_shadow_acceptance(message)
    driver_id = int(message.get("driver_id") or 0)
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    eid = str(
        message.get("location_event_id")
        or payload.get("location_event_id")
        or payload.get("tracking_event_id")
        or message.get("tracking_event_id")
        or ""
    )
    if not driver_id or not eid:
        return "comparison_unavailable"
    company_id = message.get("company_id")
    if not isinstance(company_id, int):
        company_id = (
            payload.get("company_id")
            if isinstance(payload.get("company_id"), int)
            else None
        )
    state = upsert_shadow_observation(
        driver_id=driver_id,
        location_event_id=eid,
        company_id=company_id,
        fingerprint=evaluation["shadow_fingerprint"],
        accept_status=evaluation["shadow_accept_status"],
        accept_reason=evaluation["shadow_accept_reason"],
        consumer_lag=consumer_lag,
    )
    _metric_for_state(state)
    return state


# Alias conservés pour imports existants
def record_shadow_event(message: dict[str, Any]) -> dict[str, str]:
    state = handle_shadow_raw(message)
    eid = str(message.get("location_event_id") or "")
    return {"status": state, "location_event_id": eid}


def observe_direct_for_compare(message: dict[str, Any]) -> str:
    """Hook post-UC : upsert direct (sans Kafka si déjà publié)."""
    return handle_direct_observed(message)


class TrackingShadowConsumer:
    """Boucle Kafka dual-topic + expiration lag-aware."""

    def __init__(self) -> None:
        self._consumer = None
        self._running = False
        self._initialized = False
        self._lag_high = False

    def initialize(self) -> bool:
        try:
            from kafka import KafkaConsumer
        except Exception:
            logger.exception("[shadow] kafka-python unavailable")
            return False
        self._consumer = KafkaConsumer(
            SHADOW_TOPIC,
            DIRECT_TOPIC,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
            group_id=SHADOW_GROUP,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        self._initialized = True
        return True

    def _estimate_lag(self) -> int:
        assert self._consumer is not None
        try:
            partitions = self._consumer.assignment()
            if not partitions:
                return 0
            end = self._consumer.end_offsets(list(partitions))
            lag = 0
            for tp in partitions:
                pos = self._consumer.position(tp)
                lag += max(0, int(end.get(tp, 0)) - int(pos))
            return lag
        except Exception:
            return 0

    def _commit_record(self, record: Any) -> None:
        from kafka.structs import OffsetAndMetadata, TopicPartition

        assert self._consumer is not None
        tp = TopicPartition(record.topic, record.partition)
        self._consumer.commit({tp: OffsetAndMetadata(record.offset + 1, "", -1)})

    def _sweep_expired(self, lag: int) -> None:
        was_high = self._lag_high
        self._lag_high = lag > LAG_THRESHOLD_MESSAGES
        if was_high and not self._lag_high:
            extend_deadlines_after_lag_recovery()
        for item in expire_waiting_observations(consumer_lag=lag):
            _inc_divergence(str(item.get("result") or "expired"))

    def start(self) -> None:
        if not self._initialized and not self.initialize():
            raise RuntimeError("shadow consumer init failed")
        assert self._consumer is not None
        self._running = True

        def _stop(*_args: Any) -> None:
            self._running = False

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        logger.info(
            "[shadow] start topics=%s,%s group=%s",
            SHADOW_TOPIC,
            DIRECT_TOPIC,
            SHADOW_GROUP,
        )
        while self._running:
            lag = self._estimate_lag()
            self._sweep_expired(lag)
            polled = self._consumer.poll(timeout_ms=1000)
            for _tp, records in polled.items():
                for record in records:
                    value = record.value if isinstance(record.value, dict) else {}
                    if record.topic == DIRECT_TOPIC:
                        handle_direct_observed(value, consumer_lag=lag)
                    else:
                        handle_shadow_raw(value, consumer_lag=lag)
                    self._commit_record(record)
            time.sleep(0.01)


def run_tracking_shadow_consumer() -> None:
    consumer = TrackingShadowConsumer()
    consumer.start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tracking_shadow_consumer()
