from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app = create_app()
app.app_context().push()
from models import db

start = datetime(2026, 8, 16, 13, 55, 0, tzinfo=timezone.utc)
end = datetime(2026, 8, 16, 14, 10, 0, tzinfo=timezone.utc)
did = 20135
sid_prefix = "trk_sess_1786888"

# 1) Last known session / max sequence
rows = list(db.session.execute(text("""
  SELECT tracking_session_id, MAX(sequence_id) AS max_seq, COUNT(*) AS n,
         MIN(created_at) AS first_at, MAX(created_at) AS last_at
  FROM driver_location_events
  WHERE driver_id=:did AND created_at >= :start
  GROUP BY tracking_session_id
  ORDER BY last_at DESC
"""), {"did": did, "start": start}).mappings())
print("SESSIONS")
for r in rows:
    print(dict(r))

# 2) Discover outbox / kafka related tables
tabs = list(db.session.execute(text("""
  SELECT table_name FROM information_schema.tables
  WHERE table_schema='public'
    AND (table_name ILIKE '%outbox%' OR table_name ILIKE '%tracking%event%'
         OR table_name ILIKE '%ingest%' OR table_name ILIKE '%location%queue%'
         OR table_name ILIKE '%kafka%')
  ORDER BY table_name
""")).fetchall())
print("TABLES")
for t in tabs:
    print(t[0])

# 3) Try common outbox shapes
candidates = [t[0] for t in tabs]
for tbl in candidates:
    cols = list(db.session.execute(text("""
      SELECT column_name FROM information_schema.columns
      WHERE table_schema='public' AND table_name=:t ORDER BY ordinal_position
    """), {"t": tbl}).fetchall())
    colnames = [c[0] for c in cols]
    print("COLS", tbl, colnames[:20])

# 4) Any location events for ANY driver in window after 13:57:51 (is ingest globally stuck?)
any_loc = list(db.session.execute(text("""
  SELECT driver_id, COUNT(*) AS n, MAX(created_at) AS last_at
  FROM driver_location_events
  WHERE created_at >= :cut AND created_at < :end
  GROUP BY driver_id ORDER BY n DESC LIMIT 15
"""), {"cut": datetime(2026, 8, 16, 13, 57, 51, tzinfo=timezone.utc), "end": end}).mappings())
print("ANY_LOC_AFTER_CUT")
for r in any_loc:
    print(dict(r))

# 5) Same for before cut (baseline)
any_before = list(db.session.execute(text("""
  SELECT driver_id, COUNT(*) AS n
  FROM driver_location_events
  WHERE created_at >= :start AND created_at < :cut
  GROUP BY driver_id ORDER BY n DESC LIMIT 10
"""), {"start": start, "cut": datetime(2026, 8, 16, 13, 57, 51, tzinfo=timezone.utc)}).mappings())
print("ANY_LOC_BEFORE_CUT")
for r in any_before:
    print(dict(r))
