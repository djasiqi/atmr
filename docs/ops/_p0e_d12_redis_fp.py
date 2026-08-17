"""P0-E D1/D2 — fingerprint Redis + SCAN *20135* + LOC 5882 meta (schema-aware)."""
from __future__ import annotations

import os
import re
from urllib.parse import urlparse

from app import create_app

DRIVER_ID = 20135
WITNESS_ID = 5882


def _redact_url(url: str | None) -> dict:
    if not url:
        return {"set": False}
    try:
        u = urlparse(url)
        path = (u.path or "").lstrip("/")
        db = path.split("/")[0] if path else "0"
        return {
            "set": True,
            "scheme": u.scheme,
            "host": u.hostname,
            "port": u.port,
            "db": db or "0",
            "has_password": bool(u.password),
            "has_user": bool(u.username),
        }
    except Exception as e:
        return {"set": True, "parse_error": type(e).__name__}


app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    from ext import redis_client

    print("ENV_INGEST")
    for k in (
        "TRACKING_INGEST_ASYNC_ENABLED",
        "TRACKING_PG_FIRST_CANONICAL_ENABLED",
        "TRACKING_PROCESSED_FANOUT_ENABLED",
        "KAFKA_ENABLED",
        "DRIVER_LOC_TTL_SEC",
        "REDIS_URL",
        "REDIS_HOST",
        "REDIS_PORT",
        "REDIS_DB",
        "CELERY_BROKER_URL",
    ):
        v = os.getenv(k)
        if k in ("REDIS_URL", "CELERY_BROKER_URL") and v:
            print(f"  {k}_fp={_redact_url(v)}")
        else:
            print(f"  {k}={v!r}")

    print("REDIS_CLIENT_FP")
    try:
        pool = getattr(redis_client, "connection_pool", None)
        kw = getattr(pool, "connection_kwargs", {}) if pool else {}
        print(f"  type={type(redis_client).__name__}")
        print(f"  host={kw.get('host')}")
        print(f"  port={kw.get('port')}")
        print(f"  db={kw.get('db')}")
        # ping + info server
        pong = redis_client.ping()
        print(f"  ping={pong}")
        info = redis_client.info("server")
        print(f"  redis_version={info.get('redis_version')}")
        print(f"  redis_mode={info.get('redis_mode')}")
        # which db selected
        try:
            print(f"  client_info={redis_client.client_info()}")
        except Exception:
            pass
        # write/read canary on same client (safe ephemeral)
        canary = f"p0e:d2:canary:{DRIVER_ID}"
        redis_client.setex(canary, 30, "1")
        print(f"  canary_set_ok={redis_client.get(canary) is not None}")
        redis_client.delete(canary)
    except Exception as e:
        print(f"  ERROR {type(e).__name__}: {e}")

    print("LOC_5882")
    row = db.session.execute(
        text(
            "SELECT id, driver_id, company_id, location_event_id, tracking_session_id, "
            "session_generation, sequence_id, recorded_at, created_at, location_mode, "
            "mission_id, source, payload_schema_version "
            "FROM driver_location_events WHERE id=:id"
        ),
        {"id": WITNESS_ID},
    ).mappings().fetchone()
    if row:
        for k, v in dict(row).items():
            print(f"  {k}={v}")
    else:
        print("  MISSING")

    print("SCAN_20135")
    # SCAN MATCH *20135* — capped
    cursor = 0
    found: list[str] = []
    rounds = 0
    while rounds < 50 and len(found) < 80:
        cursor, keys = redis_client.scan(cursor=cursor, match="*20135*", count=200)
        rounds += 1
        for k in keys:
            s = k.decode() if isinstance(k, bytes) else str(k)
            found.append(s)
            if len(found) >= 80:
                break
        if cursor == 0:
            break
    print(f"  rounds={rounds} count={len(found)}")
    for s in sorted(set(found))[:80]:
        print(f"  key={s}")

    # known expected keys existence
    print("EXPECTED_KEYS")
    for key in (
        f"driver:{DRIVER_ID}:loc:last_raw",
        f"driver:{DRIVER_ID}:loc:canonical",
        f"driver:{DRIVER_ID}:loc",
        f"driver:{DRIVER_ID}:active_tracking_session",
    ):
        print(f"  {key} exists={bool(redis_client.exists(key))} ttl={redis_client.ttl(key)}")

    # Latest few events source field
    print("RECENT_SOURCES")
    rows = db.session.execute(
        text(
            "SELECT id, source, sequence_id, created_at FROM driver_location_events "
            "WHERE driver_id=:d ORDER BY id DESC LIMIT 8"
        ),
        {"d": DRIVER_ID},
    ).fetchall()
    for r in rows:
        print(f"  id={r[0]} source={r[1]} seq={r[2]} created={r[3]}")
