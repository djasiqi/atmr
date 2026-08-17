from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app = create_app()
app.app_context().push()
from models import db
since = datetime.now(timezone.utc) - timedelta(minutes=25)
did = 20135
print("NOW", datetime.now(timezone.utc).isoformat())
locs = list(db.session.execute(text("""
  SELECT created_at, mission_id, recorded_at, source
  FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:since
  ORDER BY created_at
"""), {"did": did, "since": since}).fetchall())
print("LOC_N", len(locs))
for r in locs:
    print("LOC", r[0], "mission=", r[1], "rec=", r[2], "src=", r[3])
health = list(db.session.execute(text("""
  SELECT recorded_at, app_state, tracking_active, fgs_running, native_task_running,
         last_fix_age_seconds, native_last_fix_age_seconds, constraint_reason, trigger_reason
  FROM driver_device_health_events
  WHERE driver_id=:did AND recorded_at>=:since
  ORDER BY recorded_at
"""), {"did": did, "since": since}).mappings())
print("HEALTH_N", len(health))
for h in health:
    print("H", h.get("recorded_at"), "app=", h.get("app_state"), "fgs=", h.get("fgs_running"),
          "ntask=", h.get("native_task_running"), "fix_age=", h.get("last_fix_age_seconds"),
          "task_age=", h.get("native_last_fix_age_seconds"), "cstr=", h.get("constraint_reason"),
          "trig=", h.get("trigger_reason"))
