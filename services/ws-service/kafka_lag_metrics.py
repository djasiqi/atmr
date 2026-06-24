"""Métriques lag consumer Kafka ws-service (P1-1c)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

TRACKING_KAFKA_LAG_METRIC_ENABLED = (
    os.getenv("TRACKING_KAFKA_LAG_METRIC_ENABLED", "true").lower() == "true"
)
TRACKING_KAFKA_LAG_METRIC_INTERVAL_S = float(
    os.getenv("TRACKING_KAFKA_LAG_METRIC_INTERVAL_S", "15")
)

# partition key -> lag value (best-effort, exposé sur /metrics)
_lag_by_partition: dict[str, float] = {}
_fanout_emit_total = 0
_consumer_ref: Any | None = None
_group_id = "ws-service-shared"
_topic_filter: str | None = None


def configure_kafka_lag(*, group_id: str, topic: str | None = None) -> None:
    global _group_id, _topic_filter
    _group_id = group_id
    _topic_filter = topic


def register_consumer(consumer: Any | None) -> None:
    global _consumer_ref
    _consumer_ref = consumer


def record_fanout_emit() -> None:
    global _fanout_emit_total
    _fanout_emit_total += 1


def lag_snapshot() -> dict[str, float]:
    return dict(_lag_by_partition)


def fanout_emit_total() -> int:
    return _fanout_emit_total


async def publish_lag_once() -> None:
    """Calcule lag = end_offset - position pour chaque partition assignée."""
    if not TRACKING_KAFKA_LAG_METRIC_ENABLED or _consumer_ref is None:
        return
    try:
        assignment = _consumer_ref.assignment()
        if not assignment:
            return
        end_offsets = await _consumer_ref.end_offsets(list(assignment))
        for tp in assignment:
            if _topic_filter and tp.topic != _topic_filter:
                continue
            position = await _consumer_ref.position(tp)
            end_offset = end_offsets.get(tp)
            if position is None or end_offset is None:
                continue
            key = f"{_group_id}|{tp.topic}|{tp.partition}"
            _lag_by_partition[key] = max(0.0, float(end_offset - position))
    except Exception:
        logger.debug("ws-service lag metric publish failed", exc_info=True)


async def lag_publish_loop(stop_event: asyncio.Event) -> None:
    """Boucle périodique de publication du lag (P1-1c)."""
    while not stop_event.is_set():
        await publish_lag_once()
        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=TRACKING_KAFKA_LAG_METRIC_INTERVAL_S,
            )
        except TimeoutError:
            continue


def render_prometheus_lines() -> list[str]:
    """Lignes Prometheus text pour /metrics (sans dépendance prom-client)."""
    lines: list[str] = [
        "# HELP tracking_kafka_consumer_lag Lag consumer tracking (end_offset - position)",
        "# TYPE tracking_kafka_consumer_lag gauge",
    ]
    for key, lag in _lag_by_partition.items():
        group, topic, partition = key.split("|", 2)
        lines.append(
            f'tracking_kafka_consumer_lag{{group="{group}",topic="{topic}",partition="{partition}"}} {lag}'
        )
    lines.extend(
        [
            "# HELP tracking_fanout_emit_total Émissions Socket.IO driver_location depuis Kafka processed",
            "# TYPE tracking_fanout_emit_total counter",
            f'tracking_fanout_emit_total{{emitter="ws_service"}} {_fanout_emit_total}',
        ]
    )
    return lines
