from sqlalchemy import text
from app import create_app
from models import db

IDS = [
    "trk_1787339442833_72dd077",
    "trk_1787339463762_ff79938e",
    "trk_1787339486314_c7db30be",
    "trk_1787339504762_de38cfd7",
    "trk_1787339526761_45533fd5",
    "trk_1787339547763_2f8415cd",
]

app = create_app()
with app.app_context():
    b = db.session.execute(
        text("SELECT id, status::text AS status, company_id FROM booking WHERE id=51")
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
        text("SELECT last_location_event_id, is_available FROM driver WHERE id=20")
    ).mappings().first()
    print("DRIVER", dict(drv) if drv else None)
