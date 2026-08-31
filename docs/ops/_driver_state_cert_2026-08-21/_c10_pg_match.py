from sqlalchemy import text
from app import create_app
from models import db
IDS = [x for x in '''trk_1787344095291_b8e76fc0,trk_1787344116696_9cf4b0e4,trk_1787344134852_4855057b,trk_1787344159738_489b4fd4,trk_1787344181137_227701,trk_1787344202431_bcca0d1f,trk_1787344222828_230ea1eb,trk_1787344245129_349fc9b,trk_1787344266424_cdc1ac46,trk_1787344287738_e42624cf'''.split(',') if x]
app = create_app()
with app.app_context():
    ok=0; modes=set(); missions=set()
    for eid in IDS:
        row=db.session.execute(text('SELECT location_event_id, location_mode, mission_id FROM driver_location_events WHERE location_event_id=:e LIMIT 1'), {'e': eid}).mappings().first()
        if row:
            ok+=1; modes.add(str(row['location_mode'])); missions.add(str(row['mission_id'])); print('MATCH', eid, dict(row))
        else:
            print('MISS', eid)
    print(f'PG_MATCH {ok}/{len(IDS)} modes={sorted(modes)} missions={sorted(missions)}')
    drv=db.session.execute(text('SELECT last_location_event_id, is_available FROM driver WHERE id=20')).mappings().first()
    print('DRIVER', dict(drv) if drv else None)
    print('PROJECTION', bool(drv and drv['last_location_event_id'] in IDS))
    b=db.session.execute(text('SELECT id, status::text AS status FROM booking WHERE id=54')).mappings().first()
    a=db.session.execute(text('SELECT status::text AS status FROM assignment WHERE booking_id=54 ORDER BY id DESC LIMIT 1')).mappings().first()
    print('BOOKING', dict(b) if b else None, 'ASSIGNMENT', dict(a) if a else None)
