#!/usr/bin/env bash
# P0-E Phase 2 — preflight frozen gates (flag still OFF)
set -euo pipefail
cd /srv/atmr

echo "=== UTC ==="
date -u -Iseconds

echo "=== health ==="
BH=$(docker inspect -f '{{.State.Health.Status}}' atmr-backend-1)
CH=$(docker inspect -f '{{.State.Health.Status}}' atmr-tracking-kafka-consumer-1)
echo "backend=${BH} image=$(docker inspect -f '{{.Config.Image}}' atmr-backend-1)"
echo "consumer=${CH} image=$(docker inspect -f '{{.Config.Image}}' atmr-tracking-kafka-consumer-1)"
test "${BH}" = "healthy"
test "${CH}" = "healthy"

echo "=== env ==="
echo -n "consumer_OUTBOX="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX
echo -n "backend_PG_FIRST="
docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || echo UNSET
echo -n "consumer_PG_FIRST="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || echo UNSET
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX)" = "true"
test "$(docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || echo false)" = "false"
test "$(docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED || echo false)" = "false"

echo "=== PG recent driver 20135 ==="
docker exec atmr-backend-1 python - <<'PY'
from app import create_app
from sqlalchemy import text
app = create_app()
with app.app_context():
    from models import db
    rows = db.session.execute(text(
        "SELECT id, sequence_id, session_generation, left(coalesce(capture_id,''),16) AS cap, created_at "
        "FROM driver_location_events WHERE driver_id=20135 ORDER BY id DESC LIMIT 3"
    )).mappings().all()
    for r in rows:
        print(dict(r))
    d = db.session.execute(text(
        "SELECT id, last_position_at FROM drivers WHERE id=20135"
    )).mappings().first()
    print("DRIVER", dict(d) if d else None)
PY

echo "=== TB 10m ==="
BTB=$(docker logs atmr-backend-1 --since 10m 2>&1 | grep -c Traceback || true)
CTB=$(docker logs atmr-tracking-kafka-consumer-1 --since 10m 2>&1 | grep -c Traceback || true)
echo "backend_tb=${BTB} consumer_tb=${CTB}"
test "${BTB}" = "0"
test "${CTB}" = "0"

echo "=== canonical snapshot (pre) ==="
docker exec atmr-redis redis-cli HGETALL "driver:20135:loc:canonical" || true
echo -n "TTL="
docker exec atmr-redis redis-cli TTL "driver:20135:loc:canonical" || true

echo PREFLIGHT_PHASE2_PASS
