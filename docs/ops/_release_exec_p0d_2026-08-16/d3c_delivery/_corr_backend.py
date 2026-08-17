from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app = create_app()
app.app_context().push()
from models import db
# Window: HOME cycle local 15:57-16:00 = UTC 13:57-14:00
start = datetime(2026, 8, 16, 13, 57, 0, tzinfo=timezone.utc)
end = datetime(2026, 8, 16, 14, 5, 0, tzinfo=timezone.utc)
did = 20135
print("WINDOW", start.isoformat(), "->", end.isoformat())
locs = list(db.session.execute(text("""
  SELECT created_at, recorded_at, mission_id, source, sequence_id, session_generation,
         tracking_session_id, accuracy_m, raw_latitude, raw_longitude, location_event_id
  FROM driver_location_events
  WHERE driver_id=:did AND created_at >= :start AND created_at < :end
  ORDER BY created_at
"""), {"did": did, "start": start, "end": end}).mappings())
print("LOC_IN_WINDOW", len(locs))
for r in locs:
    print("LOC", r["created_at"], "rec=", r["recorded_at"], "seq=", r["sequence_id"],
          "gen=", r["session_generation"], "acc=", r["accuracy_m"],
          "lat=", round(r["raw_latitude"],5), "src=", r["source"], "sid=", (r["tracking_session_id"] or "")[:16])

# Any later LOC in next 10 min?
later = list(db.session.execute(text("""
  SELECT created_at, recorded_at, sequence_id, accuracy_m
  FROM driver_location_events
  WHERE driver_id=:did AND created_at >= :end AND created_at < :end2
  ORDER BY created_at LIMIT 20
"""), {"did": did, "end": end, "end2": end + timedelta(minutes=30)}).fetchall())
print("LOC_AFTER_WINDOW", len(later))
for r in later[:10]:
    print("LATER", r[0], "rec=", r[1], "seq=", r[2])

# HTTP request audit if table exists
for tbl in ("http_request_logs", "api_access_logs", "request_audit_logs", "audit_logs"):
    try:
        n = db.session.execute(text(f"SELECT COUNT(*) FROM {tbl} WHERE 1=0")).scalar()
        print("TABLE_OK", tbl)
    except Exception as e:
        db.session.rollback()
        print("TABLE_NO", tbl)

# Look for location-related errors in driver_device_health around window
h = list(db.session.execute(text("""
  SELECT recorded_at, app_state, fgs_running, native_task_running, last_fix_age_seconds,
         native_last_fix_age_seconds, constraint_reason, native_start_error, trigger_reason
  FROM driver_device_health_events
  WHERE driver_id=:did AND recorded_at >= :start AND recorded_at < :end2
  ORDER BY recorded_at
"""), {"did": did, "start": start, "end2": end + timedelta(minutes=30)}).mappings())
print("HEALTH_AROUND", len(h))
for r in h:
    print("H", r["recorded_at"], "app=", r["app_state"], "fgs=", r["fgs_running"],
          "ntask=", r["native_task_running"], "fix_age=", r["last_fix_age_seconds"],
          "task_age=", r["native_last_fix_age_seconds"], "cstr=", r["constraint_reason"],
          "trig=", r["trigger_reason"], "err=", (r["native_start_error"] or "")[:80])
