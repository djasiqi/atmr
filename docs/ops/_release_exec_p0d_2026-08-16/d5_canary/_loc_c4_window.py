from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
DRIVER_ID=20135
BOOKING_ID=38243
since=datetime.now(timezone.utc)-timedelta(minutes=8)
n=db.session.execute(text("SELECT count(*) FROM driver_location_events WHERE driver_id=:d AND created_at>=:s"),{"d":DRIVER_ID,"s":since}).scalar()
st=db.session.execute(text("SELECT status FROM bookings WHERE id=:b"),{"b":BOOKING_ID}).scalar()
print("LOC_8MIN", int(n or 0))
print("BOOKING_STATUS", st)