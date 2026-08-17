"""Snap DLE/canon for FLP vs Expo discriminant."""
from app import create_app
from sqlalchemy import text

app = create_app()
with app.app_context():
    from models import db
    from ext import redis_client

    a = db.session.execute(
        text(
            "SELECT tracking_session_id FROM tracking_sessions "
            "WHERE driver_id=20135 AND status='active' ORDER BY id DESC LIMIT 1"
        )
    ).scalar()
    print("SESS", a)
    if a:
        r = db.session.execute(
            text(
                "SELECT COUNT(1), COALESCE(MAX(sequence_id),0), MAX(recorded_at) "
                "FROM driver_location_events WHERE tracking_session_id=:s"
            ),
            {"s": a},
        ).first()
        print("DLE", r[0], r[1], r[2])
    raw = redis_client.hgetall("driver:20135:loc:canonical") or {}
    c = {
        (k.decode() if isinstance(k, bytes) else str(k)): (
            v.decode() if isinstance(v, bytes) else str(v)
        )
        for k, v in raw.items()
    }
    print(
        "CANON",
        c.get("sequence_id"),
        c.get("recorded_at"),
        redis_client.ttl("driver:20135:loc:canonical"),
    )
