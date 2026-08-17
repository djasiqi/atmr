#!/usr/bin/env bash
set -euo pipefail
echo "=== PUT sample 3m ==="
docker logs atmr-backend-1 --since 3m 2>&1 \
  | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin | tail -10 || true
echo "=== PUT codes 3m ==="
docker logs atmr-backend-1 --since 3m 2>&1 \
  | grep 'PUT /api/v1/driver/me/location' | grep -v Darwin \
  | awk '{print $9}' | sort | uniq -c | sort -rn || true
echo "=== consumer 3m ==="
docker logs atmr-tracking-kafka-consumer-1 --since 3m 2>&1 \
  | grep -E 'DLQ confirmed|p5b_promote|Traceback|partitions revoked' \
  | grep -v Eventlet | tail -20 || true
echo "=== PG latest ==="
docker cp /tmp/_p0e_phase2_probe.py atmr-backend-1:/tmp/_p0e_phase2_probe.py 2>/dev/null || true
docker exec atmr-backend-1 python - <<'PY'
from app import create_app
from sqlalchemy import text
app = create_app()
with app.app_context():
    from models import db
    rows = db.session.execute(text(
        "SELECT id, sequence_id, created_at, tracking_session_id "
        "FROM driver_location_events WHERE driver_id=20135 ORDER BY id DESC LIMIT 3"
    )).mappings().all()
    for r in rows:
        print(dict(r))
    d = db.session.execute(text(
        "SELECT last_position_update FROM driver WHERE id=20135"
    )).scalar()
    print("driver_last", d)
    n = db.session.execute(text(
        "SELECT COUNT(*) FROM driver_location_events "
        "WHERE driver_id=20135 AND created_at > NOW() - INTERVAL '5 minutes'"
    )).scalar()
    print("DLE_5m", n)
PY
