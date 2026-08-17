from datetime import datetime, timezone, timedelta
from sqlalchemy import text
from app import create_app
app=create_app(); app.app_context().push()
from models import db
DRIVER_ID=20135
since=datetime.now(timezone.utc)-timedelta(seconds=90)
n=db.session.execute(text('SELECT count(*) FROM driver_location_events WHERE driver_id=:d AND created_at>=:s'),{'d':DRIVER_ID,'s':since}).scalar()
print('N', int(n or 0))