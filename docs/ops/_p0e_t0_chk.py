from app import create_app
app=create_app()
with app.app_context():
 from ext import redis_client
 from sqlalchemy import text
 from models import db
 raw=redis_client.hgetall("driver:20135:loc:last_raw") or {}
 can=redis_client.hgetall("driver:20135:loc:canonical") or {}
 r=db.session.execute(text("SELECT id FROM driver_location_events WHERE driver_id=20135 ORDER BY id DESC LIMIT 1")).scalar()
 print("CHK", int(bool(raw)), int(bool(can)), int(r or 0))
