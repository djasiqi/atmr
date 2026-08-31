from sqlalchemy import text
from app import create_app
from models import db

IDS = [
    "trk_1787337224504_51c1a7c8",
    "trk_1787337245768_90e81f8c",
    "trk_1787337263770_48a5e1f4",
    "trk_1787337287770_a92a91c4",
    "trk_1787337311770_41c3c75c",
    "trk_1787337334517_9d685e4e",
    "trk_1787337356030_4049ba17",
    "trk_1787337375770_d8e7004e",
]

app = create_app()
with app.app_context():
    for eid in IDS:
        row = db.session.execute(
            text(
                "SELECT location_event_id, location_mode, mission_id, recorded_at "
                "FROM driver_location_events WHERE location_event_id=:e LIMIT 1"
            ),
            {"e": eid},
        ).mappings().first()
        print("MATCH" if row else "MISS", eid, dict(row) if row else None)
    drv = db.session.execute(
        text(
            "SELECT id, last_location_event_id, latitude, longitude, is_available "
            "FROM driver WHERE id=20"
        )
    ).mappings().first()
    print("DRIVER", dict(drv) if drv else None)
