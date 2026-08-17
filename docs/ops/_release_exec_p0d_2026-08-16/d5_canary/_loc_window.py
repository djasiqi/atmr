from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
# C1 window approx 00:58:50 - 01:11:15 Europe/Zurich = 22:58:50 - 23:11:15 UTC
start=datetime(2026,8,16,22,58,50,tzinfo=timezone.utc)
end=datetime(2026,8,16,23,12,0,tzinfo=timezone.utc)
n=db.session.execute(text("SELECT count(*) FROM driver_location_events WHERE driver_id=20135 AND created_at>=:a AND created_at<:b"),{"a":start,"b":end}).scalar()
print("LOC_C1_WINDOW", int(n or 0))
m=db.session.execute(text("SELECT id, status::text FROM booking WHERE driver_id=20135 AND status::text IN ('IN_PROGRESS','EN_ROUTE','ARRIVED') ORDER BY updated_at DESC NULLS LAST LIMIT 3")).fetchall()
print("MISSIONS", [(x[0], x[1]) for x in m])