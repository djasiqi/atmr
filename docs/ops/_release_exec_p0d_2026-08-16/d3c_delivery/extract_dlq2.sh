#!/bin/bash
set -euo pipefail
echo "=== offsets ==="
for T in driver.location.dlq.v2 driver.location.dlq.v3 driver.location.dlq driver.location.raw.v2; do
  echo "TOPIC=$T"
  docker exec atmr-kafka-broker-1 kafka-get-offsets --bootstrap-server localhost:9092 --topic "$T" || true
done
echo "=== consume dlq.v2 ==="
docker exec atmr-kafka-broker-1 kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic driver.location.dlq.v2 \
  --from-beginning \
  --timeout-ms 25000 \
  --max-messages 50 \
  > /tmp/dlq_sample.jsonl 2>/tmp/dlq_err.txt || true
echo "err:"; head -20 /tmp/dlq_err.txt || true
echo "lines=$(wc -l < /tmp/dlq_sample.jsonl | tr -d ' ')"
# If empty, try partition-specific from recent end
if [ ! -s /tmp/dlq_sample.jsonl ]; then
  echo "=== retry with property deserializer ==="
  docker exec atmr-kafka-broker-1 kafka-console-consumer \
    --bootstrap-server localhost:9092 \
    --topic driver.location.dlq.v2 \
    --partition 0 \
    --offset earliest \
    --timeout-ms 15000 \
    --max-messages 20 \
    > /tmp/dlq_sample.jsonl 2>>/tmp/dlq_err.txt || true
  echo "lines2=$(wc -l < /tmp/dlq_sample.jsonl | tr -d ' ')"
fi
head -c 500 /tmp/dlq_sample.jsonl; echo
python3 <<'PY'
import json
path='/tmp/dlq_sample.jsonl'
rows=[]
with open(path,'r',encoding='utf-8',errors='replace') as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            print('BAD_LINE', e, line[:120])
print('PARSED', len(rows))
if not rows:
    raise SystemExit(0)
# show first error types
from collections import Counter
c=Counter(str(r.get('error_type')) for r in rows)
print('error_types', c)
# pick conflict
picked=None
for o in rows:
    if 'payload_conflict' in str(o.get('error_type') or ''):
        msg=o.get('original_message') or {}
        did=msg.get('driver_id')
        if did in (20135,'20135') or True:
            picked=o
            if did in (20135,'20135'):
                break
open('/tmp/dlq_picked.json','w',encoding='utf-8').write(json.dumps(picked, indent=2, default=str)[:300000])
msg=picked.get('original_message') or {}
print('picked_driver', msg.get('driver_id'))
print('error_type', picked.get('error_type'))
print('error', str(picked.get('error'))[:400])
print('offset', picked.get('original_offset'))
print('keys', sorted(msg.keys()))
print(json.dumps(msg, indent=2, default=str)[:6000])
PY
