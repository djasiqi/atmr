from sqlalchemy import text
from app import create_app
from models import db

app = create_app()
with app.app_context():
    b = db.session.execute(
        text("SELECT id, status::text AS status FROM booking WHERE id IN (50,51)")
    ).mappings().all()
    print("BOOKINGS", [dict(x) for x in b])
    rows = db.session.execute(
        text(
            "SELECT event_id, mission_id, recorded_at, location_mode "
            "FROM driver_location_event "
            "WHERE driver_id=20 AND recorded_at > now() - interval '25 minutes' "
            "ORDER BY recorded_at DESC LIMIT 15"
        )
    ).mappings().all()
    print("PG_N", len(rows))
    for r in rows:
        print(dict(r))
