from sqlalchemy import text
from app import create_app
from models import db
app = create_app()
with app.app_context():
    b = db.session.execute(text("SELECT id, status::text AS status FROM booking WHERE id=54")).mappings().first()
    a = db.session.execute(text("SELECT id, status::text AS status FROM assignment WHERE booking_id=54 ORDER BY id DESC LIMIT 1")).mappings().first()
    print("DB_BOOKING", dict(b) if b else None)
    print("DB_ASSIGNMENT", dict(a) if a else None)
