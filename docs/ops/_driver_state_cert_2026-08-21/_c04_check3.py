from app import create_app
from models import db
from sqlalchemy import text
app = create_app()
with app.app_context():
    for eid in ["trk_1787327435533_cffcf2e7", "trk_1787327457271_8bd31a11", "trk_1787327475871_26e4220f"]:
        r = db.session.execute(text(
            "SELECT location_event_id, location_mode, mission_id, created_at FROM driver_location_events WHERE location_event_id=:e"
        ), {"e": eid}).mappings().first()
        print(eid, dict(r) if r else None)
