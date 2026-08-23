from sqlalchemy import text
from app import create_app
from models import db

IDS = [
    "trk_1787342293748_e2dd540",
    "trk_1787342315520_67bee6f3",
    "trk_1787342336927_a85983f6",
    "trk_1787342358895_a63582a2",
    "trk_1787342377747_b56c2355",
    "trk_1787342400139_d6298d6f",
]

app = create_app()
with app.app_context():
    b = db.session.execute(
        text("SELECT id, status::text AS status FROM booking WHERE id=54")
    ).mappings().first()
    a = db.session.execute(
        text(
            "SELECT id, status::text AS status FROM assignment "
            "WHERE booking_id=54 ORDER BY id DESC LIMIT 1"
        )
    ).mappings().first()
    print("BOOKING", dict(b) if b else None)
    print("ASSIGNMENT", dict(a) if a else None)
    ok = 0
    modes = set()
    missions = set()
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
            modes.add(str(row.get("location_mode")))
            missions.add(str(row.get("mission_id")))
            print("MATCH", eid, dict(row))
        else:
            print("MISS", eid)
    print(f"PG_MATCH {ok}/{len(IDS)} modes={sorted(modes)} missions={sorted(missions)}")
    drv = db.session.execute(
        text(
            "SELECT last_location_event_id, latitude, longitude, is_available "
            "FROM driver WHERE id=20"
        )
    ).mappings().first()
    print("DRIVER", dict(drv) if drv else None)
    # projection advance: last event among IDS equals or precedes driver.last
    last = drv["last_location_event_id"] if drv else None
    print("PROJECTION_IN_SET", last in IDS if last else False, "last=", last)
