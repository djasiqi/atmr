from sqlalchemy import text
from app import create_app
from models import db
IDS = ["trk_1787343400684_177717dd", "trk_1787343418742_31bf3939", "trk_1787343442742_c96b1746", "trk_1787343465094_6f4bc20", "trk_1787343486388_b794d8d6", "trk_1787343506742_eaac78ba", "trk_1787343529086_e2878568", "trk_1787343550379_7222f447"]
app = create_app()
with app.app_context():
    ok=0; modes=set(); missions=set()
    for eid in IDS:
        if not eid: continue
        row=db.session.execute(text('SELECT location_event_id, location_mode, mission_id FROM driver_location_events WHERE location_event_id=:e LIMIT 1'), {'e': eid}).mappings().first()
        if row:
            ok+=1; modes.add(str(row['location_mode'])); missions.add(str(row['mission_id'])); print('MATCH', eid, dict(row))
        else:
            print('MISS', eid)
    print(f'PG_MATCH {ok}/{len([x for x in IDS if x])} modes={sorted(modes)} missions={sorted(missions)}')
    drv=db.session.execute(text('SELECT last_location_event_id, is_available FROM driver WHERE id=20')).mappings().first()
    print('DRIVER', dict(drv) if drv else None)
    print('PROJECTION', bool(drv and drv['last_location_event_id'] in IDS))
    b=db.session.execute(text('SELECT id, status::text AS status FROM booking WHERE id=54')).mappings().first()
    a=db.session.execute(text('SELECT status::text AS status FROM assignment WHERE booking_id=54 ORDER BY id DESC LIMIT 1')).mappings().first()
    print('BOOKING', dict(b) if b else None, 'ASSIGNMENT', dict(a) if a else None)
