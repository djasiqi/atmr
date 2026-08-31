from sqlalchemy import text
from app import create_app
from models import db

IDS = [
    "trk_1787338055877_f591abee",
    "trk_1787338080767_4dce9011",
    "trk_1787338103947_42f2d3fd",
    "trk_1787338126021_3f2f87f2",
    "trk_1787338144766_9a748e1e",
    "trk_1787338168768_4cc83bad",
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
            "SELECT last_location_event_id, is_available FROM driver WHERE id=20"
        )
    ).mappings().first()
    print("DRIVER", dict(drv) if drv else None)
