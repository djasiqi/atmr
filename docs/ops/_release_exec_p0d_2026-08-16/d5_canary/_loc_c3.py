from datetime import datetime, timezone
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
start=datetime(2026,8,17,0,20,30,tzinfo=timezone.utc)
end=datetime(2026,8,17,0,26,0,tzinfo=timezone.utc)
n=db.session.execute(text("SELECT count(*) FROM driver_location_events WHERE driver_id=20135 AND created_at>=:a AND created_at<:b"),{"a":start,"b":end}).scalar()
print("LOC_C3_WINDOW", int(n or 0))
m=db.session.execute(text("SELECT id, status::text FROM booking WHERE id=38243")).fetchone()
print("MISSION", m[0] if m else None, m[1] if m else None)