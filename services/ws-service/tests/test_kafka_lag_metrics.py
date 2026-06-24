"""Tests métriques lag Kafka ws-service (P1-1c)."""

from __future__ import annotations

import kafka_lag_metrics as m


def test_render_prometheus_lines_includes_fanout_emit() -> None:
    m._fanout_emit_total = 3
    m._lag_by_partition = {"ws-service-shared|driver.location.processed.v2|0": 12.0}
    lines = m.render_prometheus_lines()
    text = "\n".join(lines)
    assert 'tracking_kafka_consumer_lag{group="ws-service-shared"' in text
    assert 'tracking_fanout_emit_total{emitter="ws_service"} 3' in text


def test_record_fanout_emit_increments() -> None:
    before = m.fanout_emit_total()
    m.record_fanout_emit()
    assert m.fanout_emit_total() == before + 1
