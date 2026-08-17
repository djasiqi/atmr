from app import create_app

app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db

    DRIVER_ID = 20135
    rows = db.session.execute(
        text(
            "SELECT id, status FROM booking WHERE driver_id=:d "
            "ORDER BY id DESC LIMIT 8"
        ),
        {"d": DRIVER_ID},
    ).fetchall()
    print("BOOKINGS")
    for r in rows:
        print(f"  id={r[0]} status={r[1]}")
    if not rows:
        print("  NONE")
    n = db.session.execute(
        text(
            "SELECT count(*) FROM driver_location_events "
            "WHERE driver_id=:d AND created_at >= now() - interval '120 seconds'"
        ),
        {"d": DRIVER_ID},
    ).scalar()
    print("LOC_120s", int(n or 0))
