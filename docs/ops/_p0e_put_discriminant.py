"""P0-E discriminant — schema-aware, lecture seule (pas de Sentry noise)."""
from __future__ import annotations

import os
from datetime import UTC, datetime

from app import create_app

DRIVER_ID = 20135


def _decode_hash(raw: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in (raw or {}).items():
        kk = k.decode() if isinstance(k, bytes) else str(k)
        vv = v.decode() if isinstance(v, bytes) else str(v)
        out[kk] = vv
    return out


app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    from ext import redis_client

    now = datetime.now(UTC)
    print("PROBE_META")
    print(f"  now_utc={now.isoformat()}")
    print(f"  driver_id={DRIVER_ID}")

    # Env flags (read-only) — canonical vs fanout
    flags = [
        "TRACKING_PG_FIRST_CANONICAL_ENABLED",
        "TRACKING_PROCESSED_FANOUT_ENABLED",
        "TRACKING_INGEST_ASYNC_ENABLED",
        "KAFKA_ENABLED",
        "DRIVER_LOC_TTL_SEC",
    ]
    print("ENV_FLAGS")
    for f in flags:
        print(f"  {f}={os.getenv(f)!r}")

    # Schema-aware: only columns that exist
    dcols = {
        r[0]
        for r in db.session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='driver'"
            )
        ).fetchall()
    }
    want = ["id", "last_position_update", "latitude", "longitude", "is_active"]
    sel = [c for c in want if c in dcols]
    drv = db.session.execute(
        text(f"SELECT {', '.join(sel)} FROM driver WHERE id=:d"),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    print("DRIVER")
    if drv:
        for k, v in dict(drv).items():
            print(f"  {k}={v}")

    latest = db.session.execute(
        text(
            "SELECT id, sequence_id, recorded_at, created_at, location_mode, "
            "mission_id, location_event_id, tracking_session_id "
            "FROM driver_location_events WHERE driver_id=:d "
            "ORDER BY id DESC LIMIT 1"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    print("LATEST_PG_LOC")
    if latest:
        for k, v in dict(latest).items():
            print(f"  {k}={v}")
        if latest["created_at"]:
            age = int((now - latest["created_at"]).total_seconds())
            print(f"  age_created_s={age}")

    for label, key in (
        ("canonical", f"driver:{DRIVER_ID}:loc:canonical"),
        ("legacy", f"driver:{DRIVER_ID}:loc"),
        ("last_raw", f"driver:{DRIVER_ID}:loc:last_raw"),
    ):
        raw = _decode_hash(redis_client.hgetall(key) or {})
        ttl = redis_client.ttl(key)
        print(f"REDIS_{label.upper()}")
        print(f"  key={key} ttl={ttl} empty={not bool(raw)}")
        if raw:
            for k in (
                "accept_status",
                "accept_reason",
                "recorded_at",
                "received_at",
                "location_mode",
                "mission_id",
                "location_event_id",
                "capture_id",
                "lat",
                "lon",
                "session_generation",
                "sequence_id",
                "source",
            ):
                if k in raw:
                    print(f"  {k}={raw[k]}")
            # dump remaining keys briefly
            extra = sorted(set(raw) - {
                "accept_status", "accept_reason", "recorded_at", "received_at",
                "location_mode", "mission_id", "location_event_id", "capture_id",
                "lat", "lon", "session_generation", "sequence_id", "source",
            })
            if extra:
                print(f"  other_keys={extra}")

    # Static verdict helper from last_raw alone
    last_raw = _decode_hash(redis_client.hgetall(f"driver:{DRIVER_ID}:loc:last_raw") or {})
    canon = _decode_hash(redis_client.hgetall(f"driver:{DRIVER_ID}:loc:canonical") or {})
    print("DISCRIMINANT_HINT")
    if not last_raw and not canon:
        print("  hint=NO_REDIS_KEYS — TTL expired or no Redis write path hit recently")
    elif last_raw and not canon:
        st = last_raw.get("accept_status", "")
        print(f"  last_raw.accept_status={st}")
        if st == "accepted_observability_only":
            print("  hint=CAS_A_LIKELY — observability-only, canonical intentionally skipped")
        elif st == "accepted_canonical":
            print("  hint=CAS_B_OR_PG_FIRST — accepted_canonical announced but key empty (writer/promote fail)")
        else:
            print(f"  hint=LAST_RAW_ONLY status={st!r}")
    elif canon:
        print("  hint=CANONICAL_PRESENT — re-check TTL / competing delete for Cas C")
    else:
        print("  hint=UNEXPECTED")
