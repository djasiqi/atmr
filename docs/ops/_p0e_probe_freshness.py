from app import create_app
from datetime import UTC, datetime

app = create_app()
with app.app_context():
    from sqlalchemy import text
    from models import db
    from services.company_driver_location_freshness import (
        last_seen_seconds_from_location_fields,
        resolve_location_freshness_timestamp,
    )
    from services.geolocation.presence import compute_location_status

    DRIVER_ID = 20135
    MISSION_ID = 38243
    WITNESS_ID = 5846  # last known from smoke window

    row = db.session.execute(
        text("SELECT * FROM driver_location_events WHERE id=:id"),
        {"id": WITNESS_ID},
    ).mappings().fetchone()
    print("WITNESS_LOC")
    for k, v in dict(row).items():
        print(f"  {k}={v}")

    recorded = row["recorded_at"]
    created = row["created_at"]  # persisted_at proxy
    # received_at not in table — created_at ≈ server ingest/persist
    loc_fields = {
        "recorded_at": recorded.isoformat() if hasattr(recorded, "isoformat") else str(recorded),
        "received_at": created.isoformat() if hasattr(created, "isoformat") else str(created),
        "ts": recorded.isoformat() if hasattr(recorded, "isoformat") else str(recorded),
    }
    now = datetime.now(UTC)
    ref = resolve_location_freshness_timestamp(loc_fields)
    age = last_seen_seconds_from_location_fields(loc_fields, now=now)
    status = compute_location_status(mode="mission_live", last_seen_seconds=age)
    print("FRESHNESS_ASOF_NOW")
    print(f"  now_utc={now.isoformat()}")
    print(f"  ref_timestamp={ref}")
    print(f"  last_seen_seconds={age}")
    print(f"  location_status={status}")
    print(f"  persist_lag_s={(created - recorded).total_seconds() if recorded and created else None}")

    # As-of just after persist (simulate freshness at ingest time)
    age_at = last_seen_seconds_from_location_fields(loc_fields, now=created)
    status_at = compute_location_status(mode="mission_live", last_seen_seconds=age_at)
    print("FRESHNESS_ASOF_PERSIST")
    print(f"  last_seen_seconds={age_at}")
    print(f"  location_status={status_at}")

    # Driver columns
    dcols = db.session.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='driver' AND column_name ILIKE '%position%' "
            "OR (table_name='driver' AND column_name ILIKE '%lat%') "
            "OR (table_name='driver' AND column_name ILIKE '%lon%') "
            "ORDER BY column_name"
        )
    ).fetchall()
    print("DRIVER_POS_COLS", [c[0] for c in dcols])

    drv = db.session.execute(
        text(
            "SELECT id, last_position_update, latitude, longitude, is_active, updated_at "
            "FROM driver WHERE id=:d"
        ),
        {"d": DRIVER_ID},
    ).mappings().fetchone()
    print("DRIVER")
    for k, v in dict(drv).items():
        print(f"  {k}={v}")
    if drv["last_position_update"]:
        lpu = drv["last_position_update"]
        if lpu.tzinfo is None:
            lpu = lpu.replace(tzinfo=UTC)
        print(f"  age_lpu_s={int((now - lpu).total_seconds())}")

    booking = db.session.execute(
        text("SELECT id, status, driver_id, updated_at FROM booking WHERE id=:m"),
        {"m": MISSION_ID},
    ).mappings().fetchone()
    print("BOOKING")
    for k, v in dict(booking).items():
        print(f"  {k}={v}")

    # Latest LOC for mission regardless of age
    latest = db.session.execute(
        text(
            "SELECT id, sequence_id, recorded_at, created_at, location_mode, "
            "tracking_session_id, mission_id "
            "FROM driver_location_events WHERE mission_id=:m ORDER BY id DESC LIMIT 1"
        ),
        {"m": MISSION_ID},
    ).mappings().fetchone()
    print("LATEST_MISSION_LOC")
    for k, v in dict(latest).items():
        print(f"  {k}={v}")

    # Redis live key
    try:
        from extensions import redis_client
        r = redis_client
        keys = []
        for pattern in (
            f"*driver*{DRIVER_ID}*loc*",
            f"*location*{DRIVER_ID}*",
            f"driver:{DRIVER_ID}:*",
            f"company:1:driver:{DRIVER_ID}:*",
        ):
            found = list(r.scan_iter(match=pattern, count=50))
            keys.extend(found[:20])
        print("REDIS_KEYS_SAMPLE", keys[:30])
        # try common keys
        for k in (
            f"driver:{DRIVER_ID}:location",
            f"driver_location:{DRIVER_ID}",
            f"company:1:drivers:locations",
            f"live:driver:{DRIVER_ID}",
        ):
            t = r.type(k)
            if t and t != b"none" and t != "none":
                print(f"REDIS_TRY {k} type={t}")
                if t in (b"hash", "hash"):
                    print(" ", r.hgetall(k))
                elif t in (b"string", "string"):
                    print(" ", r.get(k))
    except Exception as e:
        print("REDIS_ERR", type(e).__name__, e)
