#!/usr/bin/env python3
"""Analyse canary P0-D post-smoke — fenêtres temporelles autour du deploy."""
from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app

CUT = datetime(2026, 8, 16, 15, 19, 55, tzinfo=timezone.utc)  # canary deploy
HOME_START = datetime(2026, 8, 16, 15, 24, 5, tzinfo=timezone.utc)
HOME_END = datetime(2026, 8, 16, 15, 26, 20, tzinfo=timezone.utc)
LOCK_START = datetime(2026, 8, 16, 15, 26, 35, tzinfo=timezone.utc)
LOCK_END = datetime(2026, 8, 16, 15, 28, 30, tzinfo=timezone.utc)
DID = 20135

app = create_app()
app.app_context().push()
from models import db

def count_loc(a, b, label):
    n = db.session.execute(text("""
      SELECT COUNT(*) FROM driver_location_events
      WHERE driver_id=:d AND created_at >= :a AND created_at < :b
    """), {"d": DID, "a": a, "b": b}).scalar()
    print(f"LOC_{label}", n)

def max_seq(a, b, label):
    row = db.session.execute(text("""
      SELECT MIN(sequence_id), MAX(sequence_id), COUNT(*)
      FROM driver_location_events
      WHERE driver_id=:d AND created_at >= :a AND created_at < :b
    """), {"d": DID, "a": a, "b": b}).fetchone()
    print(f"SEQ_{label}", {"min": row[0], "max": row[1], "n": row[2]})

print("CUT", CUT.isoformat())
count_loc(CUT, HOME_START, "POST_DEPLOY_PRE_HOME")
count_loc(HOME_START, HOME_END, "HOME_WINDOW")
count_loc(LOCK_START, LOCK_END + timedelta(seconds=60), "LOCK_PLUS")
max_seq(CUT, datetime.now(timezone.utc), "SINCE_DEPLOY")

# last 10 LOC overall
rows = db.session.execute(text("""
  SELECT created_at, recorded_at, sequence_id, location_event_id
  FROM driver_location_events WHERE driver_id=:d
  ORDER BY created_at DESC LIMIT 12
"""), {"d": DID}).fetchall()
for r in rows:
    print("LAST", r[0], "rec", r[1], "seq", r[2], "eid", r[3])

# consumer log grep via file if any - skip
# Check if any location_event_id has duplicate rows (should be 0)
dup = db.session.execute(text("""
  SELECT location_event_id, COUNT(*) c FROM driver_location_events
  WHERE driver_id=:d AND created_at >= :cut
  GROUP BY 1 HAVING COUNT(*)>1 LIMIT 10
"""), {"d": DID, "cut": CUT}).fetchall()
print("DUP_ROWS", dup)
