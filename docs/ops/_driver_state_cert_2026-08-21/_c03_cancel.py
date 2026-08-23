from sqlalchemy import text
from app import create_app
from models import db
app = create_app()
with app.app_context():
    closed = db.session.execute(text(
        "UPDATE booking SET status='CANCELED' WHERE driver_id=20 AND status::text IN "
        "('ASSIGNED','ACCEPTED','EN_ROUTE','ARRIVED','IN_PROGRESS','PENDING') RETURNING id"
    )).fetchall()
    db.session.commit()
    print("CANCELED", [r[0] for r in closed])
