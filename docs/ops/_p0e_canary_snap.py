from app import create_app
from sqlalchemy import text

SESSION = "trk_sess_1786972692514_lauam301"
app = create_app()
with app.app_context():
    from models import db
    from ext import redis_client

    s = db.session.execute(
        text(
            "SELECT tracking_session_id, status, session_generation, updated_at "
            "FROM tracking_sessions WHERE driver_id=20135 ORDER BY id DESC LIMIT 5"
        )
    ).mappings().all()
    print("SESSIONS")
    for r in s:
        print(dict(r))

    latest = db.session.execute(
        text(
            "SELECT id, sequence_id, tracking_session_id, session_generation, "
            "mission_id, created_at FROM driver_location_events "
            "WHERE driver_id=20135 ORDER BY id DESC LIMIT 5"
        )
    ).mappings().all()
    print("LATEST_DLE")
    for r in latest:
        print(dict(r))

    on_s = db.session.execute(
        text(
            "SELECT COUNT(*), MAX(id), MAX(created_at) FROM driver_location_events "
            "WHERE driver_id=20135 AND tracking_session_id=:s"
        ),
        {"s": SESSION},
    ).first()
    print("ON_TARGET_SESSION", {"count": on_s[0], "max_id": on_s[1], "max_at": on_s[2]})

    n3 = db.session.execute(
        text(
            "SELECT COUNT(*) FROM driver_location_events "
            "WHERE driver_id=20135 AND created_at > NOW() - INTERVAL '3 minutes'"
        )
    ).scalar()
    print("DLE_3m", n3)

    raw = redis_client.hgetall("driver:20135:loc:canonical") or {}
    canon = {
        (k.decode() if isinstance(k, bytes) else k): (
            v.decode() if isinstance(v, bytes) else v
        )
        for k, v in raw.items()
    }
    print("CANON_KEYS", sorted(canon.keys()))
    print(
        "CANON",
        {
            k: canon.get(k)
            for k in (
                "tracking_session_id",
                "session_generation",
                "sequence_id",
                "location_event_id",
            )
        },
    )
    print("TTL", redis_client.ttl("driver:20135:loc:canonical"))
