#!/usr/bin/env python3
import json, os
from kafka import KafkaConsumer, TopicPartition
BOOT = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092")
EID = "trk_1786888628909_kryu2j9y"
TOPIC = "driver.location.raw.v2"
PART = 1
consumer = KafkaConsumer(
    bootstrap_servers=[x.strip() for x in BOOT.split(",") if x.strip()],
    enable_auto_commit=False,
    auto_offset_reset="earliest",
    consumer_timeout_ms=25000,
    value_deserializer=lambda b: json.loads(b.decode("utf-8")) if b else None,
)
tp = TopicPartition(TOPIC, PART)
consumer.assign([tp])
# accepted ~13:57:09 received; conflict offset 4258 — seek earlier
consumer.seek(tp, 4100)
found = []
for msg in consumer:
    if msg.offset > 4265:
        break
    val = msg.value
    if not isinstance(val, dict):
        continue
    pl = val.get("payload") if isinstance(val.get("payload"), dict) else {}
    eid = val.get("location_event_id") or pl.get("location_event_id")
    if eid == EID:
        found.append({
            "offset": msg.offset,
            "received_at_ms": val.get("received_at_ms"),
            "trace_id": val.get("trace_id"),
            "payload": pl,
        })
print("FOUND", len(found))
print(json.dumps(found, indent=2, default=str)[:50000])
open("/tmp/d4_raw_eid_versions.json","w",encoding="utf-8").write(json.dumps(found, indent=2, default=str)[:200000])
