from app import create_app
app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    d=20135
    for label, sql in [
        ("LOC_120s", "SELECT count(*) FROM driver_location_events WHERE driver_id=:d AND created_at >= now() - interval '120 seconds'"),
        ("LOC_600s", "SELECT count(*) FROM driver_location_events WHERE driver_id=:d AND created_at >= now() - interval '600 seconds'"),
        ("LOC_3600s", "SELECT count(*) FROM driver_location_events WHERE driver_id=:d AND created_at >= now() - interval '3600 seconds'"),
        ("LAST", "SELECT id, created_at, ST_Y(location::geometry) as lat, ST_X(location::geometry) as lon FROM driver_location_events WHERE driver_id=:d ORDER BY id DESC LIMIT 5"),
    ]:
        if label=="LAST":
            rows=db.session.execute(text(sql), {"d":d}).fetchall()
            print(label)
            for r in rows:
                print(" ", r)
        else:
            print(label, int(db.session.execute(text(sql), {"d":d}).scalar() or 0))
