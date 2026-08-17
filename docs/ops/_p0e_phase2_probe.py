from app import create_app
from sqlalchemy import text

app = create_app()
with app.app_context():
    from models import db
    from ext import redis_client

    rows = db.session.execute(
        text(
            "SELECT id, sequence_id, session_generation, capture_id, created_at "
            "FROM driver_location_events WHERE driver_id=20135 ORDER BY id DESC LIMIT 3"
        )
    ).mappings().all()
    print("PG_ROWS", len(rows))
    for r in rows:
        print(dict(r))
    d = db.session.execute(
        text(
            "SELECT id, company_id, last_position_update, latitude, longitude "
            "FROM driver WHERE id=20135"
        )
    ).mappings().first()
    print("DRIVER", dict(d) if d else None)

    def dec(raw):
        out = {}
        for k, v in (raw or {}).items():
            kk = k.decode() if isinstance(k, bytes) else str(k)
            vv = v.decode() if isinstance(v, bytes) else str(v)
            out[kk] = vv
        return out

    c = dec(redis_client.hgetall("driver:20135:loc:canonical") or {})
    print("CANON_KEYS", sorted(c.keys()))
    print(
        "CANON_gen",
        c.get("session_generation"),
        "seq",
        c.get("sequence_id"),
        "eid",
        c.get("location_event_id"),
    )
    print("TTL", redis_client.ttl("driver:20135:loc:canonical"))
