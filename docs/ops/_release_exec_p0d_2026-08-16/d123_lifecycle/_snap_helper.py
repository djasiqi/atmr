from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
import sys
label = sys.argv[1] if len(sys.argv) > 1 else "X"
did = int(sys.argv[2]) if len(sys.argv) > 2 else 20135
app = create_app()
app.app_context().push()
from models import db
since = datetime.now(timezone.utc) - timedelta(minutes=20)
print("LABEL", label)
print("NOW", datetime.now(timezone.utc).isoformat())
health = list(db.session.execute(text("""
  SELECT recorded_at, app_state, tracking_active, fgs_running, native_task_running,
         last_fix_age_seconds, native_last_fix_age_seconds,
         constraint_reason, native_start_error, trigger_reason, native_start_phase
  FROM driver_device_health_events
  WHERE driver_id=:did AND recorded_at>=:since
  ORDER BY recorded_at DESC LIMIT 30
"""), {"did": did, "since": since}).mappings())
print("HEALTH_N", len(health))
print("FGS_TRUE_N", sum(1 for h in health if h.get("fgs_running") is True))
print("FGS_FALSE_N", sum(1 for h in health if h.get("fgs_running") is False))
for h in health[:15]:
    print(
      "H", h.get("recorded_at"),
      "app=", h.get("app_state"),
      "trk=", h.get("tracking_active"),
      "fgs=", h.get("fgs_running"),
      "ntask=", h.get("native_task_running"),
      "fix_age=", h.get("last_fix_age_seconds"),
      "task_age=", h.get("native_last_fix_age_seconds"),
      "cstr=", h.get("constraint_reason"),
      "trig=", h.get("trigger_reason"),
      "phase=", h.get("native_start_phase"),
      "err=", (h.get("native_start_error") or "")[:100],
    )
locs = list(db.session.execute(text("""
  SELECT created_at, mission_id FROM driver_location_events
  WHERE driver_id=:did AND created_at>=:since
  ORDER BY created_at DESC LIMIT 20
"""), {"did": did, "since": since}).fetchall())
print("LOC_N", len(locs))
for r in locs[:10]:
    print("LOC", r[0], "mission=", r[1])
