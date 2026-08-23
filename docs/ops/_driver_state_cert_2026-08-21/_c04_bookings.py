from app import create_app
from models import db
from sqlalchemy import text
app = create_app()
with app.app_context():
    rows = db.session.execute(text(
        "SELECT id, status::text, driver_id, scheduled_time FROM booking "
        "WHERE driver_id=20 AND status::text IN ('ASSIGNED','EN_ROUTE','ARRIVED','IN_PROGRESS') "
        "ORDER BY id DESC LIMIT 5"
    )).mappings().all()
    print("ACTIVE", [dict(r) for r in rows])
