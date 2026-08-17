import json, os
from kafka import KafkaConsumer, TopicPartition
BOOT="kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092"
c=KafkaConsumer(bootstrap_servers=BOOT.split(","), enable_auto_commit=False, auto_offset_reset="earliest", consumer_timeout_ms=20000, value_deserializer=lambda b: json.loads(b.decode()) if b else None)
tp=TopicPartition("driver.location.raw.v2",1)
c.assign([tp])
end=c.end_offsets([tp])[tp]
start=max(0,end-80)
c.seek(tp,start)
rows=[]
for msg in c:
  if msg.offset>=end: break
  v=msg.value
  if not isinstance(v,dict): continue
  if v.get("driver_id") not in (20135,"20135"): continue
  pl=v.get("payload") or {}
  rows.append({"off":msg.offset,"recv":v.get("received_at_ms"),"eid":pl.get("location_event_id"),"seq":pl.get("sequence_id"),"rec":pl.get("recorded_at"),"sent":pl.get("sent_at")})
print("N",len(rows),"end",end)
for r in rows[-25:]:
  print(r)