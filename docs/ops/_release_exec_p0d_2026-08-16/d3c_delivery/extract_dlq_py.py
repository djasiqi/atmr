#!/usr/bin/env python3
"""Read-only: pick one DLQ event_id_payload_conflict for driver 20135 and dump fields."""
import json
import os
from kafka import KafkaConsumer, TopicPartition

BOOT = os.environ.get(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
TOPIC = os.environ.get("KAFKA_TOPIC_DRIVER_LOCATION_DLQ", "driver.location.dlq.v2")

consumer = KafkaConsumer(
    bootstrap_servers=[x.strip() for x in BOOT.split(",") if x.strip()],
    enable_auto_commit=False,
    auto_offset_reset="earliest",
    consumer_timeout_ms=20000,
    value_deserializer=lambda b: json.loads(b.decode("utf-8")) if b else None,
)
parts = consumer.partitions_for_topic(TOPIC) or set()
print("TOPIC", TOPIC, "partitions", sorted(parts), flush=True)
tps = [TopicPartition(TOPIC, p) for p in sorted(parts)]
consumer.assign(tps)
# Seek near end: last ~200 messages per partition
end = consumer.end_offsets(tps)
begin = consumer.beginning_offsets(tps)
for tp in tps:
    start = max(begin[tp], end[tp] - 200)
    consumer.seek(tp, start)
    print(f"seek {tp.partition}: {start} -> end {end[tp]}", flush=True)

picked = None
scanned = 0
matches = 0
for msg in consumer:
    scanned += 1
    val = msg.value
    if not isinstance(val, dict):
        continue
    et = str(val.get("error_type") or "")
    om = val.get("original_message") or {}
    did = om.get("driver_id")
    if did not in (20135, "20135"):
        continue
    matches += 1
    if "payload_conflict" in et:
        # prefer original_offset >= 4258
        oo = val.get("original_offset")
        if oo is None or int(oo) >= 4258:
            picked = {"dlq_meta": {
                "dlq_partition": msg.partition,
                "dlq_offset": msg.offset,
                "error_type": et,
                "error": val.get("error"),
                "original_topic": val.get("original_topic"),
                "original_partition": val.get("original_partition"),
                "original_offset": val.get("original_offset"),
                "retry_count": val.get("retry_count"),
                "timestamp": val.get("timestamp"),
            }, "original_message": om}
            break

print("SCANNED", scanned, "MATCH_20135", matches, "PICKED", bool(picked), flush=True)
if not picked:
    raise SystemExit(2)

om = picked["original_message"]
# normalize nested payload
payload = om.get("payload") if isinstance(om.get("payload"), dict) else om

def g(*keys, default=None):
    cur = om
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default

summary = {
    "location_event_id": g("location_event_id") or g("payload", "location_event_id"),
    "event_payload_hash": g("event_payload_hash") or g("payload", "event_payload_hash") or g("payload_hash"),
    "driver_id": g("driver_id"),
    "company_id": g("company_id") or g("payload", "company_id"),
    "mission_id": g("mission_id") or g("payload", "mission_id"),
    "tracking_session_id": g("tracking_session_id") or g("payload", "tracking_session_id"),
    "session_generation": g("session_generation") or g("payload", "session_generation"),
    "sequence_id": g("sequence_id") or g("payload", "sequence_id"),
    "queue_item_id": g("queue_item_id") or g("payload", "queue_item_id") or g("id"),
    "capture_id": g("capture_id") or g("payload", "capture_id"),
    "latitude": g("latitude") or g("payload", "latitude") or g("lat"),
    "longitude": g("longitude") or g("payload", "longitude") or g("lon") or g("lng"),
    "accuracy": g("accuracy") or g("payload", "accuracy") or g("accuracy_m"),
    "recorded_at": g("recorded_at") or g("payload", "recorded_at") or g("timestamp"),
    "sent_at": g("sent_at") or g("payload", "sent_at"),
    "client_created_at": g("client_created_at") or g("payload", "client_created_at"),
    "app_state": g("app_state") or g("payload", "app_state"),
    "location_mode": g("location_mode") or g("payload", "location_mode"),
    "source": g("source"),
    "received_at_ms": g("received_at_ms"),
    "top_keys": sorted(om.keys()),
}
picked["summary"] = summary
open("/tmp/d4_dlq_picked.json", "w", encoding="utf-8").write(
    json.dumps(picked, indent=2, default=str)[:400000]
)
print(json.dumps({"meta": picked["dlq_meta"], "summary": summary}, indent=2, default=str))
