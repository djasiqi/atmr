#!/usr/bin/env bash
# P0-E Phase 2 — safety gates after enable (health / TB / PG advance)
set -euo pipefail
DRIVER_ID="${1:-20135}"
BASE_ID="${2:-0}"

echo "=== UTC ==="
date -u -Iseconds

BH=$(docker inspect -f '{{.State.Health.Status}}' atmr-backend-1)
CH=$(docker inspect -f '{{.State.Health.Status}}' atmr-tracking-kafka-consumer-1)
echo "backend=${BH} consumer=${CH}"
test "${BH}" = "healthy"
test "${CH}" = "healthy"

echo -n "PG_FIRST_backend="
docker exec atmr-backend-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
echo -n "PG_FIRST_consumer="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PG_FIRST_CANONICAL_ENABLED
echo -n "OUTBOX_consumer="
docker exec atmr-tracking-kafka-consumer-1 printenv TRACKING_PERSIST_WITH_OUTBOX

BTB=$(docker logs atmr-backend-1 --since 10m 2>&1 | grep -c Traceback || true)
CTB=$(docker logs atmr-tracking-kafka-consumer-1 --since 10m 2>&1 | grep -c Traceback || true)
echo "backend_tb=${BTB} consumer_tb=${CTB}"
test "${BTB}" = "0"
test "${CTB}" = "0"

echo "=== promote logs since recreate ==="
docker logs atmr-tracking-kafka-consumer-1 --since 15m 2>&1 | grep -E 'p5b_promote|promotion canonical' | tail -20 || true

echo "=== PG latest + Driver ==="
docker exec -e DRIVER_ID="${DRIVER_ID}" -e BASE_ID="${BASE_ID}" atmr-backend-1 python - <<'PY'
import os
from app import create_app
from sqlalchemy import text
app = create_app()
did = int(os.environ["DRIVER_ID"])
base = int(os.environ.get("BASE_ID") or "0")
with app.app_context():
    from models import db
    rows = db.session.execute(text(
        "SELECT id, sequence_id, session_generation, left(coalesce(capture_id,''),16) AS cap, created_at "
        "FROM driver_location_events WHERE driver_id=:d ORDER BY id DESC LIMIT 5"
    ), {"d": did}).mappings().all()
    for r in rows:
        print(dict(r))
    if base:
        n = db.session.execute(text(
            "SELECT COUNT(*) FROM driver_location_events WHERE driver_id=:d AND id>:b"
        ), {"d": did, "b": base}).scalar()
        print(f"new_pg_rows_since_base={n}")
    d = db.session.execute(text(
        "SELECT id, last_position_update, latitude, longitude FROM driver WHERE id=:d"
    ), {"d": did}).mappings().first()
    print("DRIVER", dict(d) if d else None)
PY

echo SAFETY_GATES_OK
