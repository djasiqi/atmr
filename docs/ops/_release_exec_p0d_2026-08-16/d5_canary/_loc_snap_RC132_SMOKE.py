from app import create_app
app=create_app()
with app.app_context():
 from sqlalchemy import text
 from models import db
 n=db.session.execute(text('SELECT count(*) FROM driver_location_events WHERE driver_id=20135 AND created_at>=now()-interval ''60 seconds''')).scalar()
 print('N', int(n or 0))