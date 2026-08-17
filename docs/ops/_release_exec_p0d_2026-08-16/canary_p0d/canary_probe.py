from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
import sys, json
label = sys.argv[1]
did = int(sys.argv[2])
app = create_app(); app.app_context().push()
from models import db
now = datetime.now(timezone.utc)
since = now - timedelta(minutes=20)
cut = now  # caller passes phases; we use window
# LOC last 20m
locs = list(db.session.execute(text("""
  SELECT created_at, recorded_at, location_event_id, sequence_id, tracking_session_id,
         session_generation, event_payload_hash
  FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:since
  ORDER BY created_at DESC LIMIT 40
"""), {"did": did, "since": since}).mappings())
print("LABEL", label)
print("NOW", now.isoformat())
print("LOC_N", len(locs))
for r in locs[:15]:
    print("LOC", r["created_at"], "rec=", r["recorded_at"], "seq=", r["sequence_id"],
          "eid=", (r["location_event_id"] or "")[:28], "sid=", (r["tracking_session_id"] or "")[:24])
# ingest max seq
ing = db.session.execute(text("""
  SELECT MAX(sequence_id) AS mx, COUNT(*) AS n,
         MAX(recorded_at) AS last_rec
  FROM tracking_ingest_events
  WHERE driver_id=:did AND received_at>=:since
"""), {"did": did, "since": since}).mappings().first()
print("INGEST", dict(ing) if ing else None)
active = list(db.session.execute(text("""
  SELECT id, status::text FROM booking
  WHERE driver_id=:did AND status::text IN ('ASSIGNED','ACCEPTED','EN_ROUTE','IN_PROGRESS')
  ORDER BY id DESC LIMIT 5
"""), {"did": did}).fetchall())
print("ACTIVE", [(a[0], a[1]) for a in active])
# idempotency marker: same eid count in window
dup = db.session.execute(text("""
  SELECT location_event_id, COUNT(*) AS c
  FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:since
  GROUP BY location_event_id HAVING COUNT(*)>1
  LIMIT 5
"""), {"did": did, "since": since}).fetchall()
print("MULTI_ROW_SAME_EID", [(d[0], d[1]) for d in dup])