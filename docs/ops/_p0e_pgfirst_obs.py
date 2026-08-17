"""Surveillance courte post-activation PG_FIRST globale."""
from __future__ import annotations
import os, time
from datetime import UTC, datetime
from app import create_app
from sqlalchemy import text

DRIVER_ID = int(os.getenv("P0E_DRIVER_ID", "20135"))
WINDOW = int(os.getenv("P0E_OBS_SEC", "180"))
POLL = float(os.getenv("P0E_POLL_SEC", "15"))

def dec(raw):
    out = {}
    for k, v in (raw or {}).items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out

app = create_app()
with app.app_context():
    from models import db
    from ext import redis_client
    from services.tracking.location_candidate import is_pg_first_canonical_enabled
    from services.company_driver_locations import build_company_driver_locations_items

    print("OBS_START", datetime.now(UTC).isoformat())
    print("PG_FIRST", is_pg_first_canonical_enabled())
    if not is_pg_first_canonical_enabled():
        print("FAIL pg_first_off")
        raise SystemExit(1)

    prev_seq = None
    prev_sess = None
    samples = 0
    regressions = 0
    deadline = time.time() + WINDOW
    while time.time() < deadline:
        samples += 1
        db.session.expire_all()
        active = db.session.execute(text("""
          SELECT tracking_session_id, session_generation, status
          FROM tracking_sessions WHERE driver_id=:d AND status='active'
          ORDER BY id DESC LIMIT 1
        """), {"d": DRIVER_ID}).mappings().first()
        key = f"driver:{DRIVER_ID}:loc:canonical"
        canon = dec(redis_client.hgetall(key) or {})
        ttl = redis_client.ttl(key)
        seq = int(canon.get("sequence_id") or -1) if canon else -1
        sess = canon.get("tracking_session_id")
        print(f"SAMPLE {samples} active={dict(active) if active else None}")
        print(f"  canon_sess={sess} seq={seq} gen={canon.get('session_generation')} ttl={ttl} eid={canon.get('location_event_id')}")

        if prev_seq is not None and seq >= 0 and seq < prev_seq and sess == prev_sess:
            regressions += 1
            print("REGRESS", prev_seq, "->", seq)
        if seq >= 0:
            prev_seq = seq
            prev_sess = sess
        time.sleep(POLL)

    company_id = db.session.execute(text("SELECT company_id FROM driver WHERE id=:d"), {"d": DRIVER_ID}).scalar()
    items = build_company_driver_locations_items(int(company_id or 1), is_demo_company=False)
    hit = [i for i in items if int(i.get("driver_id") or i.get("id") or 0) == DRIVER_ID]
    print("REST", {k: hit[0].get(k) for k in ("position_source","location_status","last_seen_seconds","mission_id")} if hit else None)

    print("OBS_END", datetime.now(UTC).isoformat())
    print(f"samples={samples} regressions={regressions}")
    if regressions > 0:
        print("VERDICT OBS_FAIL_REGRESS")
        raise SystemExit(2)
    if not canon:
        print("VERDICT OBS_FAIL_NO_CANON")
        raise SystemExit(3)
    if hit and str(hit[0].get("location_status") or "") in {"last_known", "offline"}:
        print("VERDICT OBS_FAIL_REST")
        raise SystemExit(4)
    print("VERDICT OBS_PASS")
    raise SystemExit(0)
