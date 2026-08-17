#!/bin/bash
set -e
TOPIC=""
for T in driver.location.dlq.v2 driver.location.dlq.v3 driver.location.dlq; do
  echo "=== try topic $T ==="
  docker exec atmr-kafka-broker-1 bash -lc "timeout 25 kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic $T --from-beginning --timeout-ms 20000 --max-messages 120 2>/dev/null" > /tmp/dlq_sample.jsonl || true
  LINES=$(wc -l < /tmp/dlq_sample.jsonl | tr -d ' ')
  echo "lines=$LINES"
  if [ "$LINES" -gt 0 ]; then
    TOPIC=$T
    echo "TOPIC_OK=$TOPIC"
    break
  fi
done
python3 <<'PY'
import json
path="/tmp/dlq_sample.jsonl"
rows=[]
with open(path,"r",encoding="utf-8",errors="replace") as f:
    for line in f:
        line=line.strip()
        if not line:
            continue
        try:
            o=json.loads(line)
        except Exception:
            continue
        msg=o.get("original_message") or {}
        did=msg.get("driver_id")
        if did is None and isinstance(msg.get("payload"), dict):
            did=msg["payload"].get("driver_id")
        if str(did) not in ("20135",) and did != 20135:
            continue
        rows.append(o)
print("MATCH_20135", len(rows))
picked=None
for o in rows:
    off=o.get("original_offset")
    et=str(o.get("error_type") or "")
    if off is not None and int(off) >= 4258 and "payload_conflict" in et:
        picked=o
        break
if picked is None:
    for o in reversed(rows):
        if "payload_conflict" in str(o.get("error_type") or "") or "payload_conflict" in str(o.get("error") or ""):
            picked=o
            break
if picked is None and rows:
    picked=rows[-1]
print("PICKED", bool(picked))
if not picked:
    raise SystemExit(0)
open("/tmp/dlq_picked.json","w",encoding="utf-8").write(json.dumps(picked, indent=2, default=str)[:300000])
msg=picked.get("original_message") or {}
print("error_type", picked.get("error_type"))
print("error", str(picked.get("error"))[:500])
print("orig_offset", picked.get("original_offset"), "partition", picked.get("original_partition"))
print("keys_top", sorted(msg.keys()))
print(json.dumps(msg, indent=2, default=str)[:8000])
PY
