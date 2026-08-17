import json, os, time
from kafka import KafkaConsumer, TopicPartition
from collections import Counter
BOOT = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092")
TOPIC = "driver.location.dlq.v2"
DID = 20135
consumer = KafkaConsumer(
    bootstrap_servers=[x.strip() for x in BOOT.split(",") if x.strip()],
    enable_auto_commit=False,
    auto_offset_reset="earliest",
    consumer_timeout_ms=12000,
    value_deserializer=lambda b: json.loads(b.decode("utf-8")) if b else None,
)
parts = consumer.partitions_for_topic(TOPIC) or set()
tps = [TopicPartition(TOPIC, p) for p in sorted(parts)]
consumer.assign(tps)
end = consumer.end_offsets(tps)
begin = consumer.beginning_offsets(tps)
for tp in tps:
    consumer.seek(tp, max(begin[tp], end[tp] - 80))
scanned = 0
conflicts = 0
types = Counter()
recent = []
cutoff_ms = int(time.time() * 1000) - 25 * 60 * 1000
for msg in consumer:
    scanned += 1
    val = msg.value
    if not isinstance(val, dict):
        continue
    om = val.get("original_message") or {}
    if om.get("driver_id") not in (DID, str(DID)):
        continue
    et = str(val.get("error_type") or "")
    types[et] += 1
    ts = val.get("timestamp") or 0
    if "payload_conflict" in et:
        conflicts += 1
        if isinstance(ts, int) and ts >= cutoff_ms:
            pl = om.get("payload") if isinstance(om.get("payload"), dict) else {}
            recent.append({
                "dlq_offset": msg.offset,
                "error_type": et,
                "eid": om.get("location_event_id") or pl.get("location_event_id"),
                "seq": pl.get("sequence_id"),
                "recorded_at": pl.get("recorded_at"),
                "original_offset": val.get("original_offset"),
            })
print(json.dumps({"scanned": scanned, "types": dict(types), "conflicts_all_window": conflicts, "conflicts_recent": recent[:20]}, indent=2))