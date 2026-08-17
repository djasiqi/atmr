"""Lire 1-3 messages DLQ récents (diagnostic lecture seule)."""
from __future__ import annotations

import json
import os

from kafka import KafkaConsumer, TopicPartition


def main() -> None:
    brokers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
    )
    topic = os.getenv(
        "KAFKA_TOPIC_DRIVER_LOCATION_DLQ",
        os.getenv("TOPIC_DRIVER_LOCATION_DLQ", "driver.location.dlq.v2"),
    )
    consumer = KafkaConsumer(
        bootstrap_servers=brokers.split(","),
        auto_offset_reset="latest",
        enable_auto_commit=False,
        consumer_timeout_ms=8000,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    parts = consumer.partitions_for_topic(topic) or set()
    tps = [TopicPartition(topic, p) for p in sorted(parts)]
    if not tps:
        print("NO_PARTITIONS", topic)
        return
    consumer.assign(tps)
    end = consumer.end_offsets(tps)
    print("DLQ_END_OFFSETS", {f"{tp.partition}": end[tp] for tp in tps})
    for tp in tps:
        start = max(end[tp] - 5, 0)
        consumer.seek(tp, start)
    n = 0
    for msg in consumer:
        n += 1
        val = msg.value if isinstance(msg.value, dict) else {}
        orig = val.get("original_message") or {}
        payload = orig.get("payload") if isinstance(orig, dict) else {}
        if not isinstance(payload, dict):
            payload = {}
        # Prefer driver 20135 samples when present
        print("DLQ_MSG")
        print(f"  partition={msg.partition} offset={msg.offset}")
        print(f"  error_type={val.get('error_type')}")
        print(f"  error={str(val.get('error'))[:200]}")
        print(f"  driver_id={payload.get('driver_id')}")
        print(f"  location_event_id={payload.get('location_event_id')}")
        print(f"  sequence_id={payload.get('sequence_id')}")
        print(f"  session_generation={payload.get('session_generation')}")
        print(f"  tracking_session_id={payload.get('tracking_session_id')}")
        print(f"  capture_id={payload.get('capture_id')}")
        print(f"  lat={payload.get('latitude') or payload.get('lat')}")
        print(f"  lon={payload.get('longitude') or payload.get('lon')}")
        print(f"  recorded_at={payload.get('recorded_at')}")
        if n >= 12:
            break
    print(f"DLQ_SAMPLED={n}")
    consumer.close()


if __name__ == "__main__":
    main()
