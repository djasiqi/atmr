from sqlalchemy import text
from app import create_app
from models import db

IDS = [
    "trk_1787337734769_1584df9",
    "trk_1787337753769_3e7ba06e",
    "trk_1787337778770_b4d3c23e",
    "trk_1787337801407_ad2503e",
    "trk_1787337819766_d296bc9b",
]

app = create_app()
with app.app_context():
    b = db.session.execute(
        text("SELECT id, status::text AS status FROM booking WHERE id=51")
    ).mappings().first()
    print("BOOKING", dict(b) if b else None)
    ok = 0
    for eid in IDS:
        row = db.session.execute(
            text(
                "SELECT location_event_id, location_mode, mission_id "
                "FROM driver_location_events WHERE location_event_id=:e LIMIT 1"
            ),
            {"e": eid},
        ).mappings().first()
        if row:
            ok += 1
            print("MATCH", eid, dict(row))
        else:
            print("MISS", eid)
    print(f"PG_MATCH {ok}/{len(IDS)}")
    drv = db.session.execute(
        text(
            "SELECT last_location_event_id, latitude, longitude, is_available "
            "FROM driver WHERE id=20"
        )
    ).mappings().first()
    print("DRIVER", dict(drv) if drv else None)
