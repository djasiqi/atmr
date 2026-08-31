from sqlalchemy import text
from app import create_app
from models import db

app = create_app()
with app.app_context():
    last = "trk_1787344330196_209415d1"
    first = "trk_1787344095291_b8e76fc0"
    for eid in (first, last):
        row = db.session.execute(
            text(
                "SELECT location_event_id, location_mode, mission_id "
                "FROM driver_location_events WHERE location_event_id=:e LIMIT 1"
            ),
            {"e": eid},
        ).mappings().first()
        print(eid, dict(row) if row else None)
    drv = db.session.execute(
        text("SELECT last_location_event_id FROM driver WHERE id=20")
    ).mappings().first()
    print("DRIVER", dict(drv) if drv else None)
    print("PROJECTION_AVANCE", bool(drv and drv["last_location_event_id"] != first))
